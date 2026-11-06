#!/usr/bin/env python3
"""Mock Reachy Gemini DJ backend for UI work. Stdlib only.

Usage:  python3 tools/mock_api.py [port]     (default 8042)
Serves reachy_gemini_dj/static at /static/ and / (index.html), plus a fake
/api with a moving track, a bouncing mic level and accepting all POSTs.
Open http://127.0.0.1:8042/
"""
import json, math, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "reachy_gemini_dj" / "static"
T0 = time.time()
st = {
    "state": "playing", "track": "Take Five", "artist": "Dave Brubeck Quartet", "duration": 324.0,
    "base": 0.0, "since": time.time(), "queue": ["miles davis so what", "lo-fi beats"], "error": "",
    "muted": False,
    "config": {"api_key_set": True, "model": "gemini-live-2.5-flash", "voice": "Puck", "extra_instructions": "Call me Sam.",
               "music_volume": 0.8, "duck_level": 0.3, "voice_gain": 1.2, "echo_guard": 2.0, "web_search": True,
               "motion_enabled": True, "expressive": True, "dance_enabled": False},
    "events": [{"t": "12:00:01", "kind": "info", "text": "Reachy Gemini DJ starting"}],
}
VOICES = ["Puck", "Kore", "Charon", "Aoede", "Fenrir"]


def elapsed():
    if st["state"] == "playing":
        return st["base"] + time.time() - st["since"]
    return st["base"] if st["state"] == "paused" else 0.0


def ev(kind, text):
    st["events"].insert(0, {"t": time.strftime("%H:%M:%S"), "kind": kind, "text": text})


def state():
    t = time.time() - T0
    talking = int(t) % 6 < 3
    return {
        "version": "0.3.0", "gemini": "listening", "model": st["config"]["model"], "error": "",
        "muted": st["muted"], "mic_level": 0 if st["muted"] else (0.04 + 0.03 * abs(math.sin(t * 5))) if talking else 0.004,
        "heard": "play some jazz and then tell me the weather" if talking else "",
        "said": "Sure! Here is Take Five by Dave Brubeck." if int(t) % 6 >= 2 else "",
        "ffmpeg": True, "voices": VOICES, "config": st["config"],
        "music": {"state": st["state"], "track": st["track"] if st["state"] != "idle" else "", "artist": st["artist"] if st["state"] != "idle" else "",
                  "duration": st["duration"], "elapsed": min(elapsed(), st["duration"]), "queued": len(st["queue"]), "error": st["error"]},
        "google": {"connected": False, "needs_reconnect": False, "email": "", "services": [],
                   "available": {"calendar": "Calendar", "tasks": "Tasks", "gmail": "Gmail (read-only)"},
                   "client_set": False, "client_id_hint": ""},
        "motion": {"enabled": True, "busy": False, "last": "", "emotions_ready": True, "dances_ready": True, "error": ""},
        "events": st["events"][:30],
    }


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/state":
            return self.send(200, state())
        if p == "/api/robot/moves":
            return self.send(200, {"emotions": ["happy", "sad", "random"], "dances": ["groovy_sway"], "status": {}})
        f = STATIC / ("index.html" if p == "/" else p[len("/static/"):] if p.startswith("/static/") else "\0")
        if f.is_file() and STATIC in f.resolve().parents:
            ct = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}.get(f.suffix, "application/octet-stream")
            return self.send(200, f.read_bytes(), ct)
        self.send(404, b"not found", "text/plain")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        d = json.loads(self.rfile.read(n) or b"{}")
        p = self.path
        if p == "/api/music/play":
            st["queue"].append(d["query"]); ev("music", "Queued " + d["query"])
            return self.send(200, {"result": "Queued: " + d["query"], "queue_length": len(st["queue"])})
        if p == "/api/music/pause":
            st["base"] = elapsed(); st["state"] = "paused"
        elif p == "/api/music/resume":
            st["since"] = time.time(); st["state"] = "playing"
        elif p == "/api/music/skip":
            st["base"] = 0; st["since"] = time.time()
            if st["queue"]:
                st["track"], st["artist"] = st["queue"].pop(0), "Mock Artist"
            else:
                st["state"] = "idle"
        elif p == "/api/music/stop":
            st["state"] = "idle"; st["queue"].clear()
        elif p == "/api/mute":
            st["muted"] = bool(d.get("muted"))
        elif p == "/api/config":
            st["config"].update({k: v for k, v in d.items() if k != "api_key"})
        elif p == "/api/robot/move":
            return self.send(200, {"ok": True, "result": "did " + str(d)})
        elif p.startswith("/api/google/"):
            return self.send(200, {"ok": False, "error": "mock: Google not available"})
        ev("info", "POST " + p)
        self.send(200, {"ok": True})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8042
    print(f"mock on http://127.0.0.1:{port}/  (static: {STATIC})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
