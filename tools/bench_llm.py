"""Measure llama-server prompt processing and generation speed on the brain's real prompt (system + all tools).

Use against a running llama-server:
    python tools/bench_llm.py http://127.0.0.1:8089 [label]
Prints one JSON line: cold prompt eval (cache off), warm turn (cache on, new user message), tool-call turn.
"""
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build():
    from reachy_brain.config import BrainConfig
    from reachy_brain.session import BrainSession
    from reachy_gemini_dj.tool_schemas import robot_tool_specs

    class _Stub:  # minimal BrainSession without threads
        pass
    s = _Stub()
    s.cfg = BrainConfig()
    s.robot_tools = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                       "parameters": t["parameters"]}} for t in robot_tool_specs(google_services=set(), frc_enabled=False)]
    s.vision, s.gemini_key, s.allow_assist, s.extra_instructions = True, "", False, ""
    s._assist_on = lambda: False
    tools = BrainSession._tools(s)
    system = BrainSession._system(s)
    return system, tools


def call(base, msgs, tools, cache, max_tokens):
    body = {"messages": msgs, "tools": tools, "max_tokens": max_tokens, "temperature": 0.0,
            "cache_prompt": cache, "stream": False}
    t0 = time.time()
    r = httpx.post(base + "/v1/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    j = r.json()
    t = j.get("timings", {})
    m = j["choices"][0]["message"]
    return {"wall_s": round(time.time() - t0, 1), "prompt_n": t.get("prompt_n"),
            "prompt_tps": round(t.get("prompt_per_second") or 0, 1), "gen_n": t.get("predicted_n"),
            "gen_tps": round(t.get("predicted_per_second") or 0, 1),
            "out": (m.get("content") or "")[:60] or [c["function"]["name"] for c in m.get("tool_calls") or []]}


def main():
    base = sys.argv[1].rstrip("/")
    label = sys.argv[2] if len(sys.argv) > 2 else ""
    system, tools = build()
    sysmsg = {"role": "system", "content": system}
    res = {"label": label, "n_tools": len(tools)}
    res["cold"] = call(base, [sysmsg, {"role": "user", "content": "hi"}], tools, False, 1)
    res["warm_chat"] = call(base, [sysmsg, {"role": "user", "content": "Tell me a quick joke."}], tools, True, 40)
    res["warm_tool"] = call(base, [sysmsg, {"role": "user", "content": "Play some chill lofi beats."}], tools, True, 60)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
