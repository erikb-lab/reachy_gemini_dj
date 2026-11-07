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
        assert getattr(llm, "warmups", 0) == (0 if ok else 1)
        assert llm.saved == ([] if ok else llm.restored)
        assert llm.warm_key
        s.close()


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
