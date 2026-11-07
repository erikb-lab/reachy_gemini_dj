"""Fake robot client for the brain server: streams WAV files as mic audio in real time, answers tool calls with
stub results, and saves the brain's spoken replies. Use: python tools/fake_robot.py ws://127.0.0.1:8765 out_dir a.wav b.wav"""
import asyncio
import json
import sys
import time
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load16k(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a).astype(np.float32)
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


# Stub answers for robot-side tools (what the real app would return), so the brain's spoken replies are realistic
STUBS = {
    "frc_team_info": {"team": "1678", "name": "Citrus Circuits", "from": "Davis, California", "rookie_year": 2005,
                      "results_2025": [{"event": "Sacramento Regional", "qual_rank": 1, "record": "11-1-0",
                                        "playoffs": "f"}],
                      "say": "Team 1678 is Citrus Circuits, from Davis, California. They started in 2005. In 2025 at "
                             "Sacramento Regional they ranked first with a record of 11-1-0 and made it to the finals."},
    "frc_next_match": {"team": "1678", "match": "Qualification 12", "status": "Queuing soon",
                       "start": "in about 15 minutes", "alliance": "red", "partners": ["254", "971"],
                       "opponents": ["118", "148", "2056"],
                       "say": "Our next match is Qualification 12, in about 15 minutes. We're on the red alliance with "
                              "team 254, The Cheesy Poofs, 9-1-0, and team 971, Spartan Robotics, 7-3-0. Against "
                              "team 118, team 148 and team 2056."},
}


async def main(url, out, wavs):
    import base64
    import os
    import websockets
    from reachy_gemini_dj.tool_schemas import robot_tool_specs
    frame_b64 = ""
    if os.environ.get("FRAME_JPEG"):
        frame_b64 = base64.b64encode(Path(os.environ["FRAME_JPEG"]).read_bytes()).decode()
    lean = os.environ.get("LEAN") == "1"
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    # real timer and memory tools (stored in the output dir), so their spoken "say" answers are the app's own
    from reachy_gemini_dj.memory import Memory
    from reachy_gemini_dj.timers import Timers
    for p in ("timers.json", "memory.json"):
        (out / p).unlink(missing_ok=True)
    timers, memory = Timers(path=out / "timers.json"), Memory(out / "memory.json")
    pinned = [m for m in os.environ.get("MEMORY", "").split(";") if m.strip()]

    def robot_tool(name, args):
        if name in STUBS:
            return STUBS[name]
        if name in ("set_timer", "set_alarm", "list_timers", "cancel_timer", "stop_alarm"):
            return timers.handle(name, args)
        if name == "remember":
            return memory.tool_remember(args.get("fact", ""))
        if name == "forget":
            return memory.tool_forget(args.get("fact", ""))
        if name == "list_memories":
            return memory.tool_list()
        if name == "play_music":
            return {"result": f"Queued: {args.get('query', '')}"}
        return {"result": "done"}
    results = []
    async with websockets.connect(url, max_size=8 << 20) as ws:
        await ws.send(json.dumps({"type": "hello", "tools": robot_tool_specs(google_services=set(), frc_enabled=True) if lean else robot_tool_specs(), "allow_gemini_assist": False,
                                  "memory": pinned}))
        for w in wavs:
            pcm = load16k(w) + bytes(16000 * 2)          # speech + 1 s silence to close the utterance
            rec = {"wav": Path(w).name, "events": [], "tools": []}
            audio = bytearray()
            t_end = None
            first_audio = None

            async def sender():
                nonlocal t_end
                for i in range(0, len(pcm), 640):        # 20 ms frames in real time
                    await ws.send(pcm[i:i + 640])
                    await asyncio.sleep(0.02)
                t_end = time.time() - 1.0                # when the user actually stopped talking

            st = asyncio.create_task(sender())
            while True:
                msg = await asyncio.wait_for(ws.recv(), 300)
                if isinstance(msg, bytes):
                    if first_audio is None:
                        first_audio = time.time()
                    audio += msg
                    continue
                m = json.loads(msg)
                t = m.get("type")
                if t in ("heard", "said", "error", "event"):
                    rec["events"].append([t, m.get("text")])
                elif t == "tool_call":
                    rec["tools"].append([m["name"], m.get("args")])
                    res = robot_tool(m["name"], m.get("args") or {})
                    await ws.send(json.dumps({"type": "tool_result", "id": m["id"], "result": res}))
                elif t == "request_frame":
                    await ws.send(json.dumps({"type": "frame", "id": m["id"], "jpeg_b64": frame_b64}))
                elif t == "turn_done" and st.done():
                    break
            await st
            rec["first_audio_s"] = round(first_audio - t_end, 1) if first_audio and t_end else None
            rec["audio_s"] = round(len(audio) / 2 / 24000, 1)
            with wave.open(str(out / ("reply_" + Path(w).stem + ".wav")), "wb") as f:
                f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000); f.writeframes(bytes(audio))
            results.append(rec)
            print(json.dumps(rec), flush=True)
            await asyncio.sleep(1.0)
    return results


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2], sys.argv[3:]))
