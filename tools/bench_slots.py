"""Does a camera question in slot 1 keep the conversation's prompt cache in slot 0?  python tools/bench_slots.py URL JPEG"""
import base64, json, sys, time
from pathlib import Path
import httpx
sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench_llm import build

base, jpeg = sys.argv[1].rstrip("/"), sys.argv[2]
system, tools = build()
msgs = [{"role": "system", "content": system}, {"role": "user", "content": "Tell me a joke."}]


def chat(extra):
    t0 = time.time()
    r = httpx.post(base + "/v1/chat/completions", json={"messages": msgs, "tools": tools, "max_tokens": 8, **extra}, timeout=900).json()
    return round(time.time() - t0, 1), r.get("timings", {}).get("prompt_n")


def vision(extra):
    b = base64.b64encode(Path(jpeg).read_bytes()).decode()
    t0 = time.time()
    r = httpx.post(base + "/v1/chat/completions", json={"messages": [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b}}, {"type": "text", "text": "What is this?"}]}],
        "max_tokens": 40, **extra}, timeout=900).json()
    return round(time.time() - t0, 1), r.get("timings", {}).get("prompt_n"), r["choices"][0]["message"]["content"][:80]


print("chat cold      ", chat({"id_slot": 0}), flush=True)
print("chat warm      ", chat({"id_slot": 0}), flush=True)
print("vision slot 1  ", vision({"id_slot": 1}), flush=True)
print("chat after     ", chat({"id_slot": 0}), flush=True)
print("slots", [(s.get("id"), s.get("n_ctx"), s.get("is_processing")) for s in httpx.get(base + "/slots").json()] if httpx.get(base + "/slots").status_code == 200 else "no /slots")
