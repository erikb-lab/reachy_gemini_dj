"""Offline tests for the brain server: clause splitting, VAD segmentation and the turn logic with fakes."""
import threading
import time

import numpy as np
import pytest

from reachy_brain.config import BrainConfig
from reachy_brain.llm import Cancelled, split_ready
from reachy_brain.segmenter import FRAME_BYTES, Segmenter
from reachy_brain.session import BrainSession


def test_split_ready_sentence_and_clause():
    assert split_ready("Hi there.") is None                       # no whitespace after the period yet
    assert split_ready("Sure thing, here we go. More") == ("Sure thing, here we go.", "More")
    assert split_ready("Ok, ") is None                            # comma too early to break
    long = "This is a fairly long opening clause that goes on, and on"
    clause, rest = split_ready(long)
    assert clause.endswith("goes on,") and rest == "and on"


class FakeVad:
    def __init__(self, pattern):
        self.pattern = iter(pattern)

    def is_speech(self, frame, sr):
        return next(self.pattern)


def test_segmenter_cuts_utterance():
    pattern = [False] * 5 + [True] * 20 + [False] * 30
    seg = Segmenter(end_silence_ms=300, vad=FakeVad(pattern))
    events = seg.feed(b"\x00\x01" * (FRAME_BYTES // 2) * len(pattern))
    kinds = [k for k, _ in events]
    assert kinds == ["start", "utterance"]
    audio = events[1][1]
    assert audio.dtype == np.float32 and len(audio) > 20 * 480


def test_segmenter_ignores_blips():
    pattern = [True] * 3 + [False] * 40
    seg = Segmenter(end_silence_ms=300, min_speech_ms=240, vad=FakeVad(pattern))
    kinds = [k for k, _ in seg.feed(b"\x00\x00" * (FRAME_BYTES // 2) * len(pattern))]
    assert kinds == ["start", "cancel"]


class FakeLLM:
    """Scripted rounds: each is (clauses, tool_calls)."""
    def __init__(self, rounds):
        self.rounds = list(rounds)
        self.calls = []

    def chat_stream(self, msgs, tools, on_clause, cancel, tool_choice="auto", max_tokens=None):
        if max_tokens == 1:          # warm-up call
            self.warmups = getattr(self, "warmups", 0) + 1
            return "", []
        self.calls.append({"msgs": list(msgs), "tools": [t["function"]["name"] for t in tools], "choice": tool_choice})
        clauses, calls = self.rounds.pop(0)
        for c in clauses:
            if cancel.is_set():
                raise Cancelled()
            on_clause(c)
        return " ".join(clauses), [{"id": f"c{i}", "type": "function",
                                    "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)]


class FakeSTT:
    def __init__(self, text):
        self.text = text

    def transcribe(self, audio):
        return self.text


class FakeTTS:
    def stream(self, text):
        yield np.zeros(240, dtype=np.float32)


def make(rounds, heard="play some lofi", robot=None):
    sent, audio = [], []
    llm = FakeLLM(rounds)
    s = BrainSession(BrainConfig(), llm, FakeSTT(heard), FakeTTS(), sent.append, audio.append)

    def robot_side():  # answer tool calls like the robot app would
        seen = set()
        while not s._closed.is_set():
            for m in list(sent):
                if m.get("type") == "tool_call" and m["id"] not in seen:
                    seen.add(m["id"])
                    s.on_tool_result({"id": m["id"], "result": (robot or {}).get(m["name"], {"result": "done"})})
            time.sleep(0.01)
    threading.Thread(target=robot_side, daemon=True).start()
    s.on_hello({"tools": [{"name": "play_music", "description": "x", "kind": "action",
                           "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}},
                          {"name": "calendar_events", "description": "x", "kind": "info"}]})
    return s, llm, sent, audio


def run_turn(s, sent):
    s._utts.put(np.zeros(16000, dtype=np.float32))
    for _ in range(300):
        if any(m.get("type") == "turn_done" for m in sent):
            return
        time.sleep(0.01)
    raise AssertionError("turn did not finish")


def test_action_tool_after_speaking_ends_turn():
    s, llm, sent, audio = make([(["On it, spinning some lofi."], [("play_music", '{"query": "lofi"}')])])
    run_turn(s, sent)
    assert len(llm.calls) == 1                             # no second round, no repeated speech
    assert any(m.get("type") == "tool_call" and m["name"] == "play_music" for m in sent)
    assert audio and [m["text"] for m in sent if m.get("type") == "said"] == ["On it, spinning some lofi."]
    s.close()


def test_info_tool_forces_answer_without_second_lookup():
    s, llm, sent, _ = make([([], [("calendar_events", "{}")]), (["You have a dentist visit at three."], [])],
                           heard="what's on my calendar", robot={"calendar_events": {"events": ["Dentist 3pm"]}})
    run_turn(s, sent)
    assert [c["choice"] for c in llm.calls] == ["auto", "none"]
    assert "Dentist" in llm.calls[1]["msgs"][-1]["content"]
    s.close()


def test_server_tools_listed_and_gemini_only_with_key():
    s, llm, sent, _ = make([(["Hi."], [])])
    ready = next(m for m in sent if m.get("type") == "ready")
    time.sleep(0.1)
    assert getattr(llm, "warmups", 0) >= 1
    assert "web_search" in ready["tools"] and "get_weather" in ready["tools"]
    assert "ask_gemini" not in ready["tools"] and "look" not in ready["tools"]
    s.on_hello({"tools": [], "gemini_key": "k"})
    ready2 = [m for m in sent if m.get("type") == "ready"][-1]
    assert "ask_gemini" in ready2["tools"]
    s.close()


def test_history_is_kept_between_turns():
    s, llm, sent, _ = make([(["Hello!"], []), (["Again."], [])], heard="hi")
    run_turn(s, sent)
    sent.clear()
    run_turn(s, sent)
    roles = [m["role"] for m in llm.calls[1]["msgs"]]
    assert roles == ["system", "user", "assistant", "user"]
    s.close()


def test_robot_say_is_spoken_without_llm():
    s, llm, sent, audio = make([])
    s.on_say({"type": "say", "text": "Heads up! Qualification 12 is now queuing."})
    for _ in range(200):
        if any(m.get("type") == "turn_done" for m in sent):
            break
        time.sleep(0.01)
    assert [m["text"] for m in sent if m.get("type") == "said"] == ["Heads up! Qualification 12 is now queuing."]
    assert audio and llm.calls == []
    s.close()


def test_action_tool_without_speech_gets_canned_ack_and_no_second_round():
    s, llm, sent, _ = make([([], [("play_music", '{"query": "lofi"}')])])
    run_turn(s, sent)
    assert len(llm.calls) == 1
    assert [m["text"] for m in sent if m.get("type") == "said"] == ["Okay, queuing that up."]
    s.close()


def test_failed_action_tool_still_asks_the_model():
    s, llm, sent, _ = make([([], [("play_music", "{}")]), (["Sorry, that did not work."], [])],
                           robot={"play_music": {"error": "nothing found"}})
    run_turn(s, sent)
    assert len(llm.calls) == 2
    assert [m["text"] for m in sent if m.get("type") == "said"] == ["Sorry, that did not work."]
    s.close()


def test_reconnect_with_same_prompt_skips_warmup():
    s, llm, sent, _ = make([])
    time.sleep(0.2)
    assert llm.warmups == 1
    s.on_hello({"tools": s_tools(s)})
    time.sleep(0.2)
    assert llm.warmups == 1
    s.close()


def s_tools(s):
    return [{"name": t["function"]["name"], "description": t["function"]["description"], "kind": s.robot_kinds[t["function"]["name"]],
             "parameters": t["function"]["parameters"]} for t in s.robot_tools]


def test_play_request_forces_a_tool_call_first():
    s, llm, sent, _ = make([([], [("play_music", '{"query": "daft punk"}')])], heard="Please queue some daft punk.")
    run_turn(s, sent)
    assert llm.calls[0]["choice"] == "required"
    assert any(m.get("type") == "tool_call" and m["name"] == "play_music" for m in sent)
    s.close()


def test_chat_is_not_forced_to_use_a_tool():
    s, llm, sent, _ = make([(["Ha."], [])], heard="tell me a joke about a game show")
    run_turn(s, sent)
    assert llm.calls[0]["choice"] == "auto"
    s.close()


def test_ghost_transcript_is_ignored():
    s, llm, sent, _ = make([], heard="You.")
    run_turn(s, sent)
    assert llm.calls == [] and not any(m.get("type") == "heard" for m in sent)
    s.close()


def test_noise_while_thinking_does_not_cancel_the_turn_but_speech_over_reply_does():
    from types import SimpleNamespace
    from reachy_brain.session import BARGE_IN_VOICED_FRAMES
    s, llm, sent, _ = make([])
    s._busy = True
    seg = SimpleNamespace(speaking=True, voiced_frames=3, events=[("start", None)])
    seg.feed = lambda pcm: seg.events
    s.seg = seg
    s.on_audio(b"\0")
    assert not s._cancel.is_set()                 # transcribing / thinking: keep going
    s._speaking = True
    s.on_audio(b"\0")                             # speech starts over the reply ...
    assert not s._cancel.is_set()                 # ... but a short blip (its own echo) is not enough
    seg.events, seg.voiced_frames = [], BARGE_IN_VOICED_FRAMES
    s.on_audio(b"\0")
    assert s._cancel.is_set()                     # sustained speech over the spoken reply: interrupt
    s._busy = False
    s.close()


def test_word_loops_and_echo_are_ignored():
    import time
    from reachy_brain.session import echo_overlap, looks_hallucinated
    assert looks_hallucinated("Wow, cool, cool. Good. Yes, cool. Yes, cool. Yes, cool. Yes, cool.")
    assert not looks_hallucinated("Remember that our pit is A3 and our robot is Zippy.")
    said = "Team 2491 is No Mythic, from Minnesota. They started in 2008."
    assert echo_overlap("team 2491 is no mythic from minnesota", said) >= 0.6
    assert echo_overlap("what time is our next match", said) < 0.6
    s, llm, sent, _ = make([])
    now = time.monotonic()
    s._said_recent, s._play_end = [(now - 2, said)], now + 3
    assert s._reject_reason("They started in 2008, from Minnesota.", now - 1) == "echo of Reachy's own voice"
    assert s._reject_reason("They started in 2008, from Minnesota.", now + 60) == ""   # long after: a real question
    assert s._reject_reason("Yes, cool. Yes, cool. Yes, cool.", now) == "repeated words"
    assert s._reject_reason("Thanks for watching!", now) == "ghost phrase"
    assert s._reject_reason("Tell me about team 254.", now - 1) == ""
    s.close()


def test_quick_command_matching():
    from reachy_brain.session import quick_command
    assert quick_command("Please pause the music.")[0] == "pause_music"
    assert quick_command("Please play the music.")[0] == "resume_music"
    assert quick_command("Resume.")[0] == "resume_music"
    assert quick_command("Skip this song")[0] == "skip_music"
    assert quick_command("Can you stop the music?")[0] == "stop_music"
    assert quick_command("play some daft punk") is None
    assert quick_command("pause for a moment and tell me a joke") is None


def test_pause_by_voice_skips_the_llm():
    s, llm, sent, _ = make([], heard="Please pause the music.")
    s.on_hello({"tools": [{"name": "pause_music", "description": "x", "kind": "action"}]})
    run_turn(s, sent)
    assert llm.calls == []
    assert any(m.get("type") == "tool_call" and m["name"] == "pause_music" for m in sent)
    assert [m["text"] for m in sent if m.get("type") == "said"] == ["Paused."]
    s.close()


def test_forced_tool_intents():
    from reachy_brain.session import forced_tool
    allt = {"look", "get_weather", "web_search", "frc_team_info", "frc_next_match", "frc_rankings", "robot_emotion",
            "robot_dance", "play_music"}
    assert forced_tool("What do you see in front of you?", allt) == "look"
    assert forced_tool("Tell me about Team 1678.", allt) == "frc_team_info"
    assert forced_tool("Show me that you're really happy.", allt) == "robot_emotion"
    assert forced_tool("Who won the World Series this year?", allt) == "web_search"
    assert forced_tool("What's the weather like in Chicago right now?", allt) == "get_weather"
    assert forced_tool("Play some daft punk.", allt) == "play_music"
    assert forced_tool("Tell me a quick joke.", allt) is None
    assert forced_tool("What do you see?", allt - {"look"}) is None       # no camera: let the model say so


def test_history_keeps_tool_calls_so_actions_are_visible_to_the_model():
    s, llm, sent, _ = make([([], [("play_music", '{"query": "lofi"}')]), (["Hi."], [])])
    run_turn(s, sent)
    sent.clear()
    run_turn(s, sent)
    roles = [m["role"] for m in llm.calls[1]["msgs"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "user"]
    assert llm.calls[0]["choice"] == "required"
    assert "Call play_music first" in llm.calls[0]["msgs"][-1]["content"]
    assert llm.calls[1]["msgs"][1] == llm.calls[0]["msgs"][-1]        # sent exactly as before: prompt cache stays valid
    s.close()


def test_history_trims_old_turns_in_one_go():
    s, llm, sent, _ = make([])
    s.cfg.history_chars = 1000
    for i in range(30):
        s._remember([{"role": "user", "content": f"q{i} " + "x" * 80}, {"role": "assistant", "content": "a"}])
    total = sum(len(__import__("json").dumps(m)) for m in s.history)
    assert total <= 1000 and len(s.turns) >= 3
    assert s.turns[-1][0]["content"].startswith("q29")
    s.close()


def test_ask_gemini_goes_through_the_robot_when_the_robot_has_the_key():
    s, llm, sent, _ = make([([], [("ask_gemini", '{"question": "hard"}')]), (["Answer."], [])], heard="ask gemini something",
                           robot={"ask_gemini": {"answer": "42"}})
    s.on_hello({"tools": [], "assist_on_robot": True})
    ready = [m for m in sent if m.get("type") == "ready"][-1]
    assert "ask_gemini" in ready["tools"] and not s.gemini_key
    run_turn(s, sent)
    assert any(m.get("type") == "tool_call" and m["name"] == "ask_gemini" for m in sent)
    assert "42" in llm.calls[1]["msgs"][-1]["content"]
    s.close()


# ---------- say protocol, fillers, memory, slot cache, timers ----------
def add_tools(s, *specs):
    s.on_hello({"tools": [{"name": n, "description": "x", "kind": k} for n, k in specs]})


def said(sent):
    return [m["text"] for m in sent if m.get("type") == "said"]


def test_say_result_is_spoken_with_no_second_llm_round():
    s, llm, sent, _ = make([([], [("frc_next_match", "{}")])], heard="when is our next match",
                           robot={"frc_next_match": {"say": "Qualification 12 in ten minutes."}})
    add_tools(s, ("frc_next_match", "info"))
    run_turn(s, sent)
    assert len(llm.calls) == 1
    assert said(sent) == ["Checking the schedule.", "Qualification 12 in ten minutes."]
    t = s.turns[-1]
    assert [m["role"] for m in t] == ["user", "assistant", "tool", "assistant"]
    assert t[-1]["content"] == "Qualification 12 in ten minutes."
    assert "Checking" not in str(t)                                      # filler never enters history
    s.close()


def test_say_extracted_before_truncation_and_joined():
    long = "word " * 600
    s, llm, sent, _ = make([([], [("set_timer", "{}"), ("remember", "{}")])], heard="hello there",
                           robot={"set_timer": {"say": long.strip(), "pad": "x" * 100}, "remember": {"say": "Got it."}})
    add_tools(s, ("set_timer", "action"), ("remember", "action"))
    run_turn(s, sent)
    assert len(llm.calls) == 1
    assert said(sent) == [long.strip() + " Got it."]
    s.close()


def test_say_ignored_on_error_and_canned_fallback_kept():
    s, llm, sent, _ = make([([], [("set_timer", "{}")]), (["Sorry."], [])], heard="hello there",
                           robot={"set_timer": {"error": "bad", "say": "ignored"}})
    add_tools(s, ("set_timer", "action"))
    run_turn(s, sent)
    assert len(llm.calls) == 2 and said(sent) == ["Sorry."]
    s.close()
    s, llm, sent, _ = make([([], [("play_music", "{}")])], heard="hello there")
    run_turn(s, sent)
    assert said(sent) == ["Okay, queuing that up."]
    s.close()


def test_weather_say_from_server_tool(monkeypatch):
    import reachy_brain.assist as assist
    monkeypatch.setattr(assist, "get_weather", lambda **kw: {"say": "It's 70 degrees in Chicago."})
    s, llm, sent, _ = make([([], [("get_weather", '{"location": "Chicago"}')])], heard="what's the weather in Chicago")
    run_turn(s, sent)
    assert len(llm.calls) == 1
    assert said(sent) == ["Let me check the weather.", "It's 70 degrees in Chicago."]
    s.close()


def test_weather_template():
    from reachy_brain.weather import _say
    c = {"temperature_2m": 71.6, "weather_code": 2}
    d = {"temperature_2m_max": [80.2, -3.0], "temperature_2m_min": [60.0, -9.0],
         "precipitation_probability_max": [40, 0], "weather_code": [2, 71]}
    out = _say("Chicago", c, d, "fahrenheit")
    assert "72 degrees and partly cloudy" in out and "40 percent chance of rain" in out
    assert "Tomorrow looks light snow with a high of minus 3" in out
    assert not any(ch in out for ch in "°%")


def test_web_search_trimmed(monkeypatch):
    import sys
    import types
    import reachy_brain.web_search as ws

    class D:
        def __init__(self, **kw):
            pass

        def text(self, q, max_results, timelimit):
            return [{"title": "T" * 200, "body": "B" * 400, "href": "u"}] * 10
    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS=D))
    ws._cache.clear()
    ws._last_call = -100.0
    out = ws.web_search("anything", max_results=6)
    assert len(out["results"]) == 3
    assert sum(len(r["title"]) + len(r["snippet"]) for r in out["results"]) <= 520


def test_filler_resets_speaking_flag_and_is_skipped_for_quick():
    s, llm, sent, _ = make([(["It is sunny."], [])], heard="what's the weather")
    seen = {}
    orig = llm.chat_stream

    def spy(*a, **k):
        seen["speaking"] = s._speaking
        return orig(*a, **k)
    llm.chat_stream = spy
    s.on_hello({"tools": [], "gemini_key": ""})
    run_turn(s, sent)
    assert said(sent)[0] == "Let me check the weather." and seen["speaking"] is False
    assert "Let me check" not in str(llm.calls[0]["msgs"][:-1])
    s.close()
    s, llm, sent, _ = make([], heard="pause the music")
    add_tools(s, ("pause_music", "action"))
    run_turn(s, sent)
    assert said(sent) == ["Paused."]
    s.close()


def test_memory_is_pinned_after_system_and_in_every_turn():
    s, llm, sent, _ = make([(["Hi."], []), (["Again."], [])], heard="hello")
    s.on_hello({"tools": s_tools(s), "memory": ["our pit is A3", "driver is Sam"]})
    assert s.memory == ["our pit is A3", "driver is Sam"]
    run_turn(s, sent)
    sent.clear()
    run_turn(s, sent)
    for c in llm.calls:
        m = c["msgs"]
        assert m[0]["role"] == "system" and "A3" not in m[0]["content"]
        assert m[1] == {"role": "user", "content": "Things you were asked to remember: our pit is A3; driver is Sam"}
        assert m[2] == {"role": "assistant", "content": "Noted."}
    assert [m["role"] for m in llm.calls[1]["msgs"]] == ["system", "user", "assistant", "user", "assistant", "user"]
    s.close()


def test_no_memory_means_no_pinned_pair_and_memory_changes_warm_key():
    s, llm, sent, _ = make([], heard="hello")
    assert s._pinned() == []
    time.sleep(0.3)
    k1 = llm.warm_key
    s.on_hello({"tools": s_tools(s), "memory": ["a"]})
    time.sleep(0.3)
    assert llm.warm_key != k1 and llm.warmups == 2
    s.close()


def test_warm_restores_from_slot_cache_else_warms_and_saves():
    class SlotLLM(FakeLLM):
        def __init__(self, restore_ok):
            super().__init__([])
            self.restore_ok, self.restored, self.saved = restore_ok, [], []

        def slot_restore(self, f):
            self.restored.append(f)
            return self.restore_ok

        def slot_save(self, f):
            self.saved.append(f)
            return True
    for ok in (True, False):
        llm = SlotLLM(ok)
        s = BrainSession(BrainConfig(), llm, FakeSTT("x"), FakeTTS(), lambda m: None, lambda b: None)
        s.on_hello({"tools": []})
        time.sleep(0.3)
        assert llm.restored and llm.restored[0].startswith("warm-") and llm.restored[0].endswith(".bin")
        assert getattr(llm, "warmups", 0) == 1      # after a restore this only reads the remembered facts
        assert llm.saved == llm.restored
        assert llm.warm_key
        s.close()
    # the saved file is named after system prompt + tools, so new facts reuse it instead of a full warm-up
    names = []
    for facts in ([], ["our pit is A3"], ["our pit is B7", "robot is Zippy"]):
        llm = SlotLLM(True)
        s = BrainSession(BrainConfig(), llm, FakeSTT("x"), FakeTTS(), lambda m: None, lambda b: None)
        s.on_hello({"tools": [], "memory": facts})
        time.sleep(0.3)
        names.append(llm.restored[0])
        s.close()
    assert len(set(names)) == 1


def test_warm_survives_slot_errors():
    class Bad(FakeLLM):
        def slot_restore(self, f):
            raise RuntimeError("boom")

        def slot_save(self, f):
            raise RuntimeError("boom")
    llm = Bad([])
    s = BrainSession(BrainConfig(), llm, FakeSTT("x"), FakeTTS(), lambda m: None, lambda b: None)
    s.on_hello({"tools": []})
    time.sleep(0.3)
    assert llm.warmups == 1 and llm.warm_key
    s.close()


def test_llama_server_slot_calls_and_pruning(tmp_path, monkeypatch):
    import os
    import reachy_brain.llm as L
    cfg = BrainConfig(slot_cache_dir=str(tmp_path), llama_url="http://x")
    srv = L.LlamaServer(cfg)
    calls = []

    class R:
        status_code = 200
    monkeypatch.setattr(L.httpx, "post", lambda url, **kw: calls.append((url, kw)) or R())
    assert srv.slot_save("warm-a.bin") and srv.slot_restore("warm-a.bin")
    assert calls[0][0].endswith("/slots/0") and calls[0][1]["params"] == {"action": "save"}
    assert calls[1][1]["params"] == {"action": "restore"} and calls[1][1]["json"] == {"filename": "warm-a.bin"}

    def boom(*a, **k):
        raise L.httpx.ConnectError("no")
    monkeypatch.setattr(L.httpx, "post", boom)
    assert srv.slot_restore("x") is False
    srv.proc = object()                          # pretend we launched it: the local dir gets pruned
    for i in range(5):
        p = tmp_path / f"warm-{i}.bin"
        p.write_text("x")
        os.utime(p, (1000 + i, 1000 + i))
    srv._prune_slots()
    assert sorted(f.name for f in tmp_path.iterdir()) == ["warm-2.bin", "warm-3.bin", "warm-4.bin"]


def test_slot_cache_dir_flag():
    assert BrainConfig.from_args(["--slot-cache-dir", "/x/y"]).slot_cache_dir == "/x/y"
    assert "/tmp" not in BrainConfig().slot_cache_dir


def test_new_intents():
    from reachy_brain.session import forced_tool
    allt = {"set_timer", "set_alarm", "list_timers", "cancel_timer", "stop_alarm", "remember", "forget", "list_memories",
            "web_search", "get_weather", "play_music"}

    def f(t):
        return forced_tool(t, allt)
    assert f("Set a timer for the pasta") == "set_timer"
    assert f("remind me in 5 minutes") == "set_timer"
    assert f("set an alarm for 7 am") == "set_alarm" and f("wake me at six") == "set_alarm"
    assert f("remember that our pit is A3") == "remember"
    assert f("don't forget the battery swap") == "remember"
    assert f("forget about the pit") == "forget"
    assert f("what do you remember") == "list_memories"
    assert f("how long is left on my timer") == "list_timers" and f("show my timers") == "list_timers"
    assert f("cancel the timer") == "cancel_timer" and f("cancel my alarm") == "cancel_timer"
    assert f("stop the alarm") == "stop_alarm"
    assert f("what time is it") is None
    assert f("remember that the latest news is good") == "remember"          # not web_search
    assert forced_tool("set a timer", {"web_search"}) is None


def test_quick_timer_commands():
    from reachy_brain.session import quick_command
    assert quick_command("Set a timer for 10 minutes.") == ("set_timer", "Done.", {"duration": "10 minutes"})
    assert quick_command("start a timer for two hours and thirty minutes")[2] == {"duration": "two hours and thirty minutes"}
    assert quick_command("set a 5 minute timer")[2] == {"duration": "5 minute"}
    assert quick_command("10 minutes timer")[2] == {"duration": "10 minutes"}
    assert quick_command("set a timer for 10 minutes for the pasta") is None
    assert quick_command("Cancel the timer.") == ("cancel_timer", "Done.", {"which": ""})
    assert quick_command("cancel my timers")[2] == {"which": "all"}
    assert quick_command("Turn off the alarm")[0] == "stop_alarm"
    assert quick_command("stop the alarm")[0] == "stop_alarm"
    assert quick_command("what time is it") is None


def test_quick_timer_passes_args_and_speaks_say():
    s, llm, sent, _ = make([], heard="Set a timer for 10 minutes.",
                           robot={"set_timer": {"say": "Timer set for 10 minutes."}})
    add_tools(s, ("set_timer", "action"))
    run_turn(s, sent)
    assert llm.calls == []
    call = next(m for m in sent if m.get("type") == "tool_call")
    assert call["name"] == "set_timer" and call["args"] == {"duration": "10 minutes"}
    assert said(sent) == ["Timer set for 10 minutes."]
    assert '"duration": "10 minutes"' in s.turns[-1][1]["tool_calls"][0]["function"]["arguments"]
    s.close()


def test_unsent_tool_results_are_stubbed_in_history():
    """A result answered by its "say" never reaches the LLM, so history keeps a stub, not the whole result."""
    from reachy_brain.session import _compact, _compact_unsent
    msgs = [{"role": "tool", "tool_call_id": "a", "content": '{"big": "' + "x" * 900 + '", "say": "hi"}'}]
    _compact_unsent(msgs, [(0, {"big": "x" * 900, "say": "hi"})])
    assert msgs[0] == {"role": "tool", "tool_call_id": "a", "content": '{"ok": true}'}
    assert "nope" in _compact({"error": "nope"})


# ---------- v0.7 shortcuts: music queue/volume/favorites/seek, play args, local answers, quiet mode ----------
def test_music_shortcuts_map_to_tools_and_args():
    from reachy_brain.session import quick_command as q
    cases = {
        "What's playing?": ("music_queue", {"action": "now_playing"}),
        "what song is this": ("music_queue", {"action": "now_playing"}),
        "What is this song?": ("music_queue", {"action": "now_playing"}),
        "who sings this": ("music_queue", {"action": "now_playing"}),
        "what's next": ("music_queue", {"action": "list"}),
        "What's in the queue?": ("music_queue", {"action": "list"}),
        "what's up next": ("music_queue", {"action": "list"}),
        "clear the queue": ("music_queue", {"action": "clear"}),
        "shuffle the queue": ("music_queue", {"action": "shuffle"}),
        "shuffle the music": ("music_queue", {"action": "shuffle"}),
        "shuffle it": ("music_queue", {"action": "shuffle"}),
        "louder": ("adjust_volume", {"direction": "up"}),
        "Turn it up!": ("adjust_volume", {"direction": "up"}),
        "volume up": ("adjust_volume", {"direction": "up"}),
        "a bit louder": ("adjust_volume", {"direction": "up"}),
        "quieter": ("adjust_volume", {"direction": "down"}),
        "turn it down": ("adjust_volume", {"direction": "down"}),
        "volume down": ("adjust_volume", {"direction": "down"}),
        "softer": ("adjust_volume", {"direction": "down"}),
        "save this song": ("favorite_song", {"action": "save"}),
        "I like this song": ("favorite_song", {"action": "save"}),
        "add this to my favorites": ("favorite_song", {"action": "save"}),
        "favorite this": ("favorite_song", {"action": "save"}),
        "play my favorites": ("favorite_song", {"action": "play"}),
        "keep the music going": ("music_queue", {"action": "autoplay_on"}),
        "stop after this song": ("music_queue", {"action": "autoplay_off"}),
        "play the next song": ("skip_music", {}),
        "skip to the next song": ("skip_music", {}),
        "skip": ("skip_music", {}),
        "Please pause the music.": ("pause_music", {}),
    }
    for heard, (tool, args) in cases.items():
        got = q(heard)
        assert got and got[0] == tool and got[2] == args, (heard, got)


def test_music_shortcuts_do_not_steal_real_requests():
    from reachy_brain.session import quick_command as q, local_answer as la
    for heard in ["what's next on my calendar", "what's playing at the cinema", "what's next for team 254",
                  "turn it up to eleven", "shuffle the cards", "clear the table",
                  "I like this song a lot, who is it", "play my favorite song by queen", "add this to my shopping list",
                  "turn the lights up", "help me remember my locker code", "what's the date of the next match",
                  "play some louder music", "what song is this from"]:
        assert q(heard) is None, heard
        assert la(heard, {"play_music"}) is None, heard
    assert q("keep going")[0] == "resume_music"                      # existing behaviour kept


def test_music_shortcut_only_fires_when_tool_is_advertised_and_speaks_say():
    s, llm, sent, _ = make([(["It is quiet."], [])], heard="What's playing?")
    run_turn(s, sent)                                  # no music_queue tool: the model gets it
    assert len(llm.calls) == 1
    s.close()
    s, llm, sent, _ = make([], heard="what's playing", robot={"music_queue": {"say": "Daft Punk, Get Lucky."}})
    add_tools(s, ("music_queue", "info"))
    run_turn(s, sent)
    call = next(m for m in sent if m.get("type") == "tool_call")
    assert llm.calls == [] and call["name"] == "music_queue" and call["args"] == {"action": "now_playing"}
    assert said(sent) == ["Daft Punk, Get Lucky."]
    assert s.turns[-1][2]["content"] == '{"ok": true}'                # stored as a stub, not the result
    s.close()
    s, llm, sent, _ = make([], heard="turn it up", robot={"adjust_volume": {"ok": True}})
    add_tools(s, ("adjust_volume", "action"))
    run_turn(s, sent)
    assert llm.calls == [] and said(sent) == ["Louder."]              # fallback text when there is no say
    s.close()


def test_play_args_parsed_without_llm():
    from reachy_brain.session import quick_command as q
    assert q("play daft punk next") == ("play_music", "Okay, queuing that up.", {"query": "daft punk", "when": "next"})
    assert q("Play get lucky now.")[2] == {"query": "get lucky", "when": "now"}
    assert q("play get lucky right now")[2] == {"query": "get lucky", "when": "now"}
    assert q("queue up 3 jazz songs")[2] == {"query": "jazz", "count": 3}
    assert q("queue up three songs by daft punk")[2] == {"query": "daft punk", "count": 3}
    assert q("play 5 songs by queen")[2] == {"query": "queen", "count": 5}
    assert q("play 50 songs by queen") is None and q("play 0 songs by queen") is None
    assert q("play it now") is None and q("play music next") is None and q("play some daft punk") is None
    assert q("play the game now") is None


def test_play_intents_and_hints():
    from reachy_brain.session import forced_tool, play_hint
    allt = {"play_music", "web_search", "frc_next_match", "set_timer"}
    for heard in ["add thriller to the queue", "queue up some abba", "play the latest hits next",
                  "play something by queen now", "play 3 songs from the eighties", "play three songs by abba"]:
        assert forced_tool(heard, allt) == "play_music", heard
    assert forced_tool("when do we play next", allt) != "play_music"
    assert forced_tool("add a timer to the queue", allt) != "play_music"
    assert forced_tool("play the queue now", {"web_search"}) is None
    assert play_hint("play thriller next") == '; use when="next"'
    assert play_hint("play thriller now") == '; use when="now"'
    assert play_hint("add thriller to the queue") == '; use when="last"'
    assert play_hint("play 3 songs by abba") == "; use count=3"
    assert play_hint("play three songs by abba") == "; use count=3"
    assert play_hint("play some abba") == ""


def test_play_hint_reaches_the_llm():
    s, llm, sent, _ = make([([], [("play_music", '{"query": "thriller", "when": "next"}')])],
                           heard="add thriller to the queue next")
    run_turn(s, sent)
    assert llm.calls[0]["choice"] == "required"
    assert 'Call play_music first; use when="next". Do not answer from memory.' in llm.calls[0]["msgs"][-1]["content"]
    s.close()


def test_play_next_by_voice_skips_the_llm_and_passes_args():
    s, llm, sent, _ = make([], heard="play daft punk next", robot={"play_music": {"say": "Daft Punk is up next."}})
    run_turn(s, sent)
    call = next(m for m in sent if m.get("type") == "tool_call")
    assert llm.calls == [] and call["args"] == {"query": "daft punk", "when": "next"}
    assert said(sent) == ["Daft Punk is up next."]
    s.close()


def test_seek_shortcuts():
    from reachy_brain.session import quick_command as q

    def to(h):
        got = q(h)
        return got[2]["to"] if got and got[0] == "seek_music" else got
    assert to("skip to 1:30") == "1:30" and to("Skip to 1.30.") == "1:30" and to("go to 1:15") == "1:15"
    assert to("skip to one thirty") == "1:30" and to("jump to two fifteen") == "2:15"
    assert to("jump to 2 minutes") == "2 minutes" and to("jump to two minutes") == "2 minutes"
    assert to("go to 90 seconds") == "90 seconds" and to("skip to 1 30") == "1:30"
    assert to("skip to the beginning") == "start"
    assert to("skip ahead 30 seconds") == "+30" and to("go forward 30 seconds") == "+30"
    assert to("fast forward thirty seconds") == "+30" and to("fast forward") == "+10"
    assert to("skip ahead a minute") == "+60" and to("skip forward") == "+10"
    assert to("go back 10 seconds") == "-10" and to("rewind 15 seconds") == "-15" and to("rewind a bit") == "-10"
    assert to("rewind") == "-10"
    assert to("start the song over") == "start" and to("restart the song") == "start" and to("from the beginning") == "start"
    assert q("skip")[0] == "skip_music"
    assert q("skip to the next song")[0] == "skip_music"
    for neg in ["go back to my calendar", "skip to the end of my homework", "go to 7",
                "go back to the first slide", "rewind the story of team 254", "skip ahead to next week",
                "go to 25:99"]:
        got = q(neg)
        assert got is None or got[0] != "seek_music", (neg, got)


def test_seek_only_when_advertised_and_speaks_say():
    s, llm, sent, _ = make([(["Sorry."], [])], heard="skip to 1:30")
    run_turn(s, sent)
    assert len(llm.calls) == 1 and not any(m.get("type") == "tool_call" for m in sent)
    s.close()
    s, llm, sent, _ = make([], heard="skip to 1:30", robot={"seek_music": {"say": "Jumped to 1:30."}})
    add_tools(s, ("seek_music", "action"))
    run_turn(s, sent)
    call = next(m for m in sent if m.get("type") == "tool_call")
    assert llm.calls == [] and call["name"] == "seek_music" and call["args"] == {"to": "1:30"}
    assert said(sent) == ["Jumped to 1:30."]
    s.close()


def test_local_answers():
    import datetime
    from reachy_brain.session import local_answer as la, _HELP_AREAS
    now = datetime.datetime(2026, 11, 6, 21, 42)
    assert la("What time is it?", set(), now=now) == ("It's 9:42 PM.", True)
    assert la("what's the time", set(), now=now)[0] == "It's 9:42 PM."
    assert la("what time is it", set(), now=datetime.datetime(2026, 11, 6, 0, 5))[0] == "It's 12:05 AM."
    assert la("What's the date?", set(), now=now)[0] == "It's Friday, November 6."
    assert la("what day is it", set(), now=now)[0] == "It's Friday, November 6."
    assert la("say that again", set(), "Sunny and 70.") == ("Sunny and 70.", False)
    for p in ["repeat that", "what did you say", "come again"]:
        assert la(p, set(), "Hello there.")[0] == "Hello there."
    assert la("repeat that", set()) == ("I haven't said anything yet.", False)
    for p in ["never mind", "Nevermind.", "cancel", "forget it", "nothing"]:
        assert la(p, set()) == ("Okay.", False), p
    h = la("what can you do", {"play_music", "robot_emotion", "get_weather", "set_timer", "remember", "frc_rankings"})[0]
    assert h == "I can help with music, moves and emotions, weather and web, timers and alarms, memory, and FRC. Just ask!"
    assert la("help", {"look"})[0] == "I can help with camera. Just ask!"
    assert la("help", set())[0].startswith("I can chat")
    assert "camera" not in la("what are your features", {"play_music"})[0]
    full = la("help", {t for _, ns in _HELP_AREAS for t in ns})[0]
    assert len(full.split()) < 35
    for neg in ["help me remember my locker code", "remember that I need help", "forget about the pit",
                "cancel the timer", "cancel my alarm", "what time is the next match", "what's the date of the final",
                "what day is the competition", "nothing else matters by metallica", "never mind the music, play jazz",
                "say that again in french", "what time is it in tokyo", "can you help me with my homework"]:
        assert la(neg, {"remember", "play_music"}) is None, neg


def test_local_answers_skip_llm_and_are_not_stored():
    s, llm, sent, _ = make([], heard="What time is it?")
    run_turn(s, sent)
    assert llm.calls == [] and s.turns == []
    assert said(sent)[0].startswith("It's ") and said(sent)[0].endswith("M.")
    s.close()
    # "say that again" repeats the last real answer, not a filler; "never mind" says Okay and keeps it
    s, llm, sent, _ = make([(["It is sunny."], [])], heard="what's the weather")
    run_turn(s, sent)
    assert said(sent) == ["Let me check the weather.", "It is sunny."] and s._last_said == "It is sunny."
    s.stt.text = "say that again"
    sent.clear()
    run_turn(s, sent)
    assert said(sent) == ["It is sunny."] and len(llm.calls) == 1
    s.stt.text = "never mind"
    sent.clear()
    run_turn(s, sent)
    assert said(sent) == ["Okay."] and s._last_said == "It is sunny." and len(s.turns) == 1
    s.close()
    s, llm, sent, _ = make([], heard="come again")
    run_turn(s, sent)
    assert said(sent) == ["I haven't said anything yet."] and llm.calls == []
    s.close()


def test_help_me_remember_still_goes_to_remember():
    from reachy_brain.session import forced_tool
    assert forced_tool("remember that I need help", {"remember", "web_search"}) == "remember"
    s, llm, sent, _ = make([([], [("remember", '{"fact": "locker 12"}')])], heard="help me remember that my locker is 12")
    add_tools(s, ("remember", "action"))
    run_turn(s, sent)
    assert "Call remember first" in llm.calls[0]["msgs"][-1]["content"]   # went to the model, not the help shortcut
    s.close()


def test_quiet_mode_shortcuts():
    from reachy_brain.session import quick_command as q
    assert q("go to sleep") == ("quiet_mode", "Okay, quiet for 30 minutes.", {"minutes": 30})
    assert q("Stop listening.")[2] == {"minutes": 30} and q("take a break")[2] == {"minutes": 30}
    assert q("be quiet for 10 minutes")[2] == {"minutes": 10}
    assert q("be quiet for ten minutes")[2] == {"minutes": 10}
    assert q("go to sleep for an hour")[2] == {"minutes": 60} and q("be quiet for half an hour")[2] == {"minutes": 30}
    assert q("be quiet for 2 hours")[2] == {"minutes": 120}
    for neg in ["be quiet", "go to sleep early tonight", "take a break from the music and tell me a joke",
                "stop listening to rock music", "remind me to take a break"]:
        got = q(neg)
        assert got is None or got[0] != "quiet_mode", (neg, got)
    s, llm, sent, _ = make([], heard="be quiet for 5 minutes", robot={"quiet_mode": {"say": "Quiet for 5 minutes."}})
    add_tools(s, ("quiet_mode", "action"))
    run_turn(s, sent)
    call = next(m for m in sent if m.get("type") == "tool_call")
    assert llm.calls == [] and call["args"] == {"minutes": 5} and said(sent) == ["Quiet for 5 minutes."]
    s.close()


def test_frc_followups_force_team_info_only_with_team_in_history():
    from reachy_brain.session import forced_tool
    allt = {"frc_team_info", "web_search", "play_music", "get_weather"}
    ctx = "Tell me about team 2491\nTeam 2491 is NorthStar Robotics from Minnesota."
    for heard in ["I'm asking about their Duluth competition.", "How did they do at Champs in Houston?",
                  "Please look up all of their information as much as you can.",
                  "I would like more information on their 2026 season.", "how did we do at the district championship",
                  "show me all of their history"]:
        assert forced_tool(heard, allt, ctx) == "frc_team_info", heard
    assert forced_tool("Tell me about team 2491 Duluth competition this year.", allt) == "frc_team_info"
    assert forced_tool("How did they do at the game", allt, "we talked about the weather") is None
    assert forced_tool("How did they do at the game", allt) is None
    assert forced_tool("how did they do at Champs", allt - {"frc_team_info"}, ctx) != "frc_team_info"
    assert forced_tool("their 2026 season", allt, "play some music") is None


def test_frc_hint_names_args_and_follows_history():
    s, llm, sent, _ = make([([], [("frc_team_info", "{}")]), (["Ok."], []), ([], [("frc_team_info", "{}")]), (["Ok."], [])],
                           heard="Tell me about team 2491 at Duluth")
    add_tools(s, ("frc_team_info", "info"))
    run_turn(s, sent)
    assert '(Call frc_team_info first; set event, year or detail="all" if asked. Do not answer from memory.)' in \
        llm.calls[0]["msgs"][-1]["content"]
    s.stt.text = "How did they do at Champs in Houston?"
    sent.clear()
    run_turn(s, sent)
    assert llm.calls[2]["choice"] == "required"
    assert "take the team from the conversation" in llm.calls[2]["msgs"][-1]["content"]
    s.close()


def test_cue_and_q_spellings_and_counts():
    from reachy_brain.session import quick_command as q, forced_tool, play_hint
    assert q("Please cue 10 Minecraft songs.") == ("play_music", "Okay, queuing that up.", {"query": "minecraft", "count": 10})
    assert q("queue 10 minecraft songs")[2] == {"query": "minecraft", "count": 10}
    assert q("Q up 3 jazz songs")[2] == {"query": "jazz", "count": 3}
    assert q("cue up three songs by daft punk")[2] == {"query": "daft punk", "count": 3}
    allt = {"play_music", "web_search"}
    assert forced_tool("Please cue some abba", allt) == "play_music"
    assert forced_tool("add thriller to the cue", allt) == "play_music"
    assert forced_tool("That's only one, I wanted ten songs.", allt) == "play_music"
    assert play_hint("That's only one, I wanted ten songs.") == "; use count=10"
    assert forced_tool("I wanted a pizza", allt) is None
