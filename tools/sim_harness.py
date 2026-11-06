"""Run the real app against the Reachy Mini simulator with WAV files as the microphone (no mic/speaker needed).

  reachy-mini-daemon --sim --headless --no-media          # terminal 1 (simulator)
  reachy-brain --model ... --mmproj ...                    # terminal 2 (local brain)
  python tools/sim_harness.py out_dir [--image cam.jpg] a.wav b.wav ...

The app's web UI is served on http://127.0.0.1:8042 while it runs. Speech the robot "plays" is saved to
out_dir/robot_speech.wav; music goes to a silent SDL audio driver.
"""
import argparse
import json
import os
import sys
import threading
import time
import wave
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")      # keep ffplay music silent
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class FakeMedia:
    """Stands in for mini.media: 16 kHz mic fed from WAVs in real time, 24 kHz mono speaker captured."""

    def __init__(self, image=None):
        self.queue = bytearray()
        self.lock = threading.Lock()
        self.out = []
        self.t_next = time.monotonic()
        self.image = image

    def get_input_audio_samplerate(self): return 16000
    def get_output_audio_samplerate(self): return 24000
    def get_output_channels(self): return 1
    def start_recording(self): pass
    def start_playing(self): pass
    def stop_recording(self): pass
    def stop_playing(self): pass

    def feed(self, pcm16: bytes):
        with self.lock:
            self.queue += pcm16

    def get_audio_sample(self):
        now = time.monotonic()
        if now < self.t_next:
            time.sleep(self.t_next - now)
        self.t_next = max(self.t_next + 0.02, time.monotonic() - 0.1)
        with self.lock:
            chunk, self.queue[:] = bytes(self.queue[:640]), self.queue[640:]
        chunk = chunk.ljust(640, b"\0")
        return np.frombuffer(chunk, dtype="<i2").astype(np.float32) / 32768.0

    def push_audio_sample(self, data):
        self.out.append(np.asarray(data, dtype=np.float32).reshape(-1))
        time.sleep(len(self.out[-1]) / 24000 * 0.98)    # play in real time like a speaker

    def get_frame(self):
        return self.image


class SimRobot:
    def __init__(self, mini, media):
        self._mini, self.media = mini, media

    def __getattr__(self, k):
        return getattr(self._mini, k)


def load16k(path):
    import soundfile as sf
    a, sr = sf.read(path, dtype="float32")
    if a.ndim > 1:
        a = a.mean(axis=1)
    if sr != 16000:
        a = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / sr)), np.arange(len(a)), a).astype(np.float32)
    return (np.clip(a, -1, 1) * 32767).astype("<i2").tobytes()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("wavs", nargs="+")
    ap.add_argument("--image")
    ap.add_argument("--brain-url", default="ws://127.0.0.1:8765")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    os.environ["HOME"] = str(out / "home")                  # isolated app config, never the user's real one
    from reachy_mini import ReachyMini
    import uvicorn
    from reachy_gemini_dj.main import ReachyGeminiDJ

    img = None
    if a.image:
        from PIL import Image
        img = np.asarray(Image.open(a.image).convert("RGB"))[..., ::-1].copy()   # BGR like a camera
    media = FakeMedia(img)
    app = ReachyGeminiDJ()
    app.config.backend, app.config.brain_url, app.config.api_key = "local", a.brain_url, ""
    app.config.dance_enabled = True
    srv = uvicorn.Server(uvicorn.Config(app.settings_app, host="127.0.0.1", port=8042, log_level="warning"))
    threading.Thread(target=srv.run, daemon=True).start()
    stop = threading.Event()
    report = []
    with ReachyMini(host="localhost", connection_mode="localhost_only", media_backend="no_media") as mini:
        robot = SimRobot(mini, media)
        th = threading.Thread(target=app.run, args=(robot, stop), daemon=True)
        th.start()
        for _ in range(120):
            if app.local_session.connected and app.local_session.info:
                break
            time.sleep(0.5)
        print("connected:", app.local_session.connected, "backend:", app.voice.active_name, flush=True)
        for w in a.wavs:
            app.heard_text = app.said_text = ""
            n_events = len(app.events)
            pose0 = mini.get_current_head_pose().copy()
            t0 = time.time()
            media.feed(load16k(w) + bytes(32000))
            t_end = t0 + len(load16k(w)) / 32000
            first = None
            n_out = len(media.out)
            idle_since = None
            while time.time() - t0 < 240:
                if first is None and len(media.out) > n_out:
                    first = time.time() - t_end
                busy = app.local_session.state not in ("listening",) or app.audio_pipe._is_robot_speaking \
                    or bool(media.queue)
                if not busy and app.said_text:
                    idle_since = idle_since or time.time()
                    if time.time() - idle_since > 2.5:
                        break
                else:
                    idle_since = None
                time.sleep(0.1)
            moved = float(np.abs(mini.get_current_head_pose() - pose0).max())
            evs = [e for e in list(app.events)[: len(app.events) - n_events]]
            rec = {"wav": Path(w).name, "heard": app.heard_text, "said": app.said_text,
                   "first_audio_s": round(first, 1) if first else None,
                   "events": [e["text"] for e in reversed(evs)], "head_moved": round(moved, 3),
                   "music": app.music.get_status().get("state")}
            report.append(rec)
            print(json.dumps(rec), flush=True)
        st = app.music.stop()
        stop.set()
        th.join(10)
    speech = np.concatenate(media.out) if media.out else np.zeros(1, np.float32)
    with wave.open(str(out / "robot_speech.wav"), "wb") as f:
        f.setnchannels(1); f.setsampwidth(2); f.setframerate(24000)
        f.writeframes((np.clip(speech, -1, 1) * 32767).astype("<i2").tobytes())
    (out / "report.json").write_text(json.dumps(report, indent=1))
    srv.should_exit = True


if __name__ == "__main__":
    main()
