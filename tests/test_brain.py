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
    s, llm, sent, _ = make([])
    s._busy = True
    s.seg.feed = lambda pcm: [("start", None)]
    s.on_audio(b"\0")
    assert not s._cancel.is_set()                 # transcribing / thinking: keep going
    s._speaking = True
    s.on_audio(b"\0")
    assert s._cancel.is_set()                     # user talks over the spoken reply: interrupt
    s._busy = False
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
