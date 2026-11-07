"""Long-term memory: store, dedupe/update, limits, forget, persistence, say strings, and what the models are given."""
import json
import logging
from types import SimpleNamespace

import pytest

from reachy_gemini_dj import memory as M
from reachy_gemini_dj.gemini import GeminiSession, build_tools
from reachy_gemini_dj.memory import Memory
from reachy_gemini_dj.remote_session import RemoteSession
from reachy_gemini_dj.tool_schemas import INFO_TOOLS, robot_tool_specs


@pytest.fixture
def mem(tmp_path):
    return Memory(tmp_path / "memory.json")


def test_add_and_clean(mem):
    r = mem.add("  The   team's pit\nis A3  ")
    assert r["status"] == "added" and mem.texts() == ["The team's pit is A3"]
    assert mem.add("   ")["status"] == "empty"
    mem.add("x" * 500)
    assert len(mem.texts()[-1]) == M.MAX_CHARS


def test_duplicates_ignored(mem):
    mem.add("Our pit is A3")
    assert mem.add("our pit is a3.")["status"] == "duplicate"
    assert mem.add("OUR  PIT IS  A3!!")["status"] == "duplicate"
    assert len(mem) == 1


def test_update_heuristic_replaces_same_subject(mem):
    mem.add("our pit is A3")
    mem.add("my name is Sam")
    r = mem.add("our pit is now B7")
    assert r["status"] == "updated" and r["old"] == "our pit is A3"
    assert sorted(mem.texts()) == ["my name is Sam", "our pit is now B7"]
    assert len(mem) == 2
    # possessive "team's" says the same thing as "our"
    assert mem.add("The team's pit is C1")["status"] == "updated"
    # a different thing is not replaced
    assert mem.add("my dog's name is Rex")["status"] == "added"
    assert mem.add("my name is Alex")["status"] == "updated"
    assert "my dog's name is Rex" in mem.texts()
    assert M.subject_key("I like jazz") == ""          # not an "X is Y" fact: never replaces anything


def test_limits(mem):
    for i in range(M.MAX_FACTS):
        assert mem.add(f"unique fact number {i} about topic{i}")["status"] == "added"
    assert mem.add("one more thing entirely")["status"] == "full"
    out = mem.tool_remember("one more thing entirely")
    assert "error" in out and "full" in out["say"]
    assert len(mem) == M.MAX_FACTS


def test_persistence_and_atomic_file(tmp_path):
    p = tmp_path / "sub" / "memory.json"
    a = Memory(p)
    a.add("our pit is A3")
    a.add("my name is Sam")
    b = Memory(p)
    assert b.texts() == a.texts() and [f["id"] for f in b.listing()] == [f["id"] for f in a.listing()]
    assert not list(p.parent.glob("*.tmp"))
    data = json.loads(p.read_text())
    assert set(data["facts"][0]) == {"id", "text", "created"}
    p.write_text("not json{")
    assert Memory(p).texts() == []


def test_delete_id_and_clear(mem):
    mem.add("our pit is A3")
    fid = mem.listing()[0]["id"]
    assert mem.delete_id("nope") is False and mem.delete_id(fid) is True and len(mem) == 0
    mem.add("a b c")
    assert mem.clear() == 1 and len(mem) == 0


def test_forget_fuzzy_single_match(mem):
    mem.add("The team's pit is A3")
    mem.add("My name is Sam")
    out = mem.tool_forget("the pit location")
    assert out["say"] == "Okay, I forgot that." and mem.texts() == ["My name is Sam"]


def test_forget_ambiguous_does_not_delete(mem):
    mem.add("The pit is A3")
    mem.add("The pit crew captain is Dana")
    out = mem.tool_forget("pit")
    assert out["say"].startswith("Which one:") and "A3" in out["say"] and "Dana" in out["say"]
    assert len(mem) == 2
    assert mem.tool_forget("the pit is A3")["say"] == "Okay, I forgot that."      # exact text wins
    assert mem.texts() == ["The pit crew captain is Dana"]


def test_forget_no_match_and_empty(mem):
    mem.add("The pit is A3")
    assert "don't remember" in mem.tool_forget("favorite pizza")["say"] and len(mem) == 1
    assert "error" in mem.tool_forget("  ")


def test_forget_all_requires_the_word_all(mem):
    for t in ("The pit is A3", "My name is Sam", "Bob likes tea"):
        mem.add(t)
    assert mem.tool_forget("the pit")["say"] == "Okay, I forgot that." and len(mem) == 2
    out = mem.tool_forget("everything")
    assert out["say"] == "Okay, I forgot everything." and len(mem) == 0
    mem.add("The pit is A3")
    assert mem.tool_forget("all of it")["say"] == "Okay, I forgot everything." and len(mem) == 0
    mem.add("Our all star is Sam")
    assert len(mem) == 1 and mem.tool_forget("all star")["say"] == "Okay, I forgot that."
    assert "wasn't remembering" in mem.tool_forget("all")["say"]


def test_say_strings(mem):
    assert mem.tool_remember("The team's pit is A3")["say"] == "Got it, I'll remember that the team's pit is A3."
    assert mem.tool_remember("The team's pit is A3")["say"] == "I already know that."
    assert mem.tool_remember("our pit is now B7")["say"].startswith("Okay, I updated that:")
    assert "error" in mem.tool_remember("")
    assert mem.tool_remember("NASA is cool")["say"].endswith("that NASA is cool.")
    mem.clear()
    assert "not remembering anything" in mem.tool_list()["say"]
    mem.add("The pit is A3")
    assert mem.tool_list()["say"] == "I remember one thing: The pit is A3."
    mem.add("My name is Sam")
    mem.add("Bob likes tea.")
    out = mem.tool_list()
    assert out["say"] == "I remember 3 things: The pit is A3; My name is Sam; Bob likes tea." and out["count"] == 3
    for i in range(10):
        mem.add(f"filler fact {i} zz{i}")
    assert "and 5 more." in mem.tool_list()["say"]


def test_fact_text_not_logged(mem, caplog):
    with caplog.at_level(logging.DEBUG):
        mem.add("The secret code is PLUGH")
        mem.tool_forget("secret code")
    assert "PLUGH" not in caplog.text and "secret" not in caplog.text


# ---- tools / declarations ----
def test_tool_declarations_and_info_tools():
    names = {d.name for t in build_tools() for d in t.function_declarations}
    assert {"remember", "forget", "list_memories"} <= names
    assert "list_memories" in INFO_TOOLS
    specs = {s["name"]: s for s in robot_tool_specs()}
    assert specs["remember"]["kind"] == "action" and specs["forget"]["kind"] == "action"
    assert specs["list_memories"]["kind"] == "info"
    assert specs["remember"]["parameters"]["required"] == ["fact"]


def _session(mem):
    s = GeminiSession(SimpleNamespace(api_key="k", voice="Puck", extra_instructions="", web_search=False,
                                      model="m"), None, None, None, lambda k, t: None)
    s.memory = mem
    return s


def test_execute_tool_dispatch(mem, caplog):
    s = _session(mem)
    assert "remember that your pit is A3" in s.execute_tool("remember", {"fact": "your pit is A3"})["say"]
    assert "I remember one thing" in s.execute_tool("list_memories", {})["say"]
    assert s.execute_tool("forget", {"fact": "pit"})["say"] == "Okay, I forgot that."
    s.memory = None
    assert "error" in s.execute_tool("remember", {"fact": "x"})


def test_gemini_system_instruction_contains_memory(mem):
    s = _session(mem)
    base = s._build_live_config().system_instruction.parts[0].text
    assert "Things you were asked to remember" not in base
    mem.add("The team's pit is A3")
    mem.add("My name is Sam")
    text = s._build_live_config().system_instruction.parts[0].text
    assert "Things you were asked to remember: The team's pit is A3; My name is Sam." in text


def test_system_prompt_is_plain():
    from reachy_gemini_dj.gemini import SYSTEM_PROMPT
    assert "\n1." not in SYSTEM_PROMPT and "Capabilities:" not in SYSTEM_PROMPT and "- " not in SYSTEM_PROMPT
    assert "remember" in SYSTEM_PROMPT and "timers" in SYSTEM_PROMPT


# ---- hello to the local brain ----
def _remote(mem=None):
    cfg = SimpleNamespace(api_key="", extra_instructions="", allow_gemini_assist=False, brain_url="ws://x")
    r = RemoteSession(cfg, execute_tool=lambda n, a: {}, on_event=lambda k, t: None, tool_specs=lambda: [])
    if mem is not None:
        r.get_memory = mem.texts
    return r


def test_hello_contains_memory(mem):
    assert _remote()._hello()["memory"] == []
    mem.add("The team's pit is A3")
    hello = _remote(mem)._hello()
    assert hello["memory"] == ["The team's pit is A3"]
    assert {"type", "client", "tools", "extra_instructions", "allow_gemini_assist", "assist_on_robot"} <= set(hello)
