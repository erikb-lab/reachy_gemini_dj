"""Tiny mock of the Reachy Gemini DJ API for testing static/robot.js and settings.js.

Run from anywhere:  python tools/mock_api2.py [port]   (default 8765)
Open http://127.0.0.1:8765/  (serves tools/panel_test.html and /static/*).
Failure injection: POST /api/config with brain_url == "bad" returns HTTP 500.
Set env MOCK_NEWSTATE=1 to include active_backend/brain in /api/state.
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "reachy_gemini_dj" / "static"
HARNESS = Path(__file__).resolve().parent / "panel_test.html"

STATE = {
    "version": "0.3.0", "gemini": "connected", "model": "m", "error": "", "muted": False,
    "voices": ["Puck", "Kore", "Charon", "Aoede"],
    "config": {
        "api_key_set": True, "model": "gemini-2.5-flash-native-audio-latest", "voice": "Puck",
        "extra_instructions": "", "music_volume": 1.0, "duck_level": 0.2, "voice_gain": 1.5,
        "echo_guard": 1.5, "web_search": False, "motion_enabled": True, "expressive": True,
        "dance_enabled": False,
    },
    "motion": {"enabled": True, "busy": False, "last": "", "emotions_ready": True, "dances_ready": True, "error": ""},
    "events": [],
}
if os.environ.get("MOCK_NEWSTATE"):
    STATE["active_backend"] = "gemini"
    STATE["brain"] = {"ok": False, "error": "connection refused"}
EMOTIONS = ["random", "happy", "sad", "surprised", "thinking", "yes", "no", "welcoming"]
DANCES = ["groovy_sway", "head_bop", "robot_wave"]
TYPES = {".html": "text/html", ".js": "text/javascript", ".css": "text/css"}


class H(BaseHTTPRequestHandler):
    def _json(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/api/state":
            return self._json(STATE)
        if p == "/api/robot/moves":
            return self._json({"emotions": EMOTIONS, "dances": DANCES, "status": STATE["motion"]})
        f = HARNESS if p in ("/", "/index.html") else STATIC / p[len("/static/"):] if p.startswith("/static/") else None
        if f and f.is_file():
            b = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", TYPES.get(f.suffix, "application/octet-stream"))
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        else:
            self.send_error(404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        d = json.loads(self.rfile.read(n) or b"{}")
        print("POST", self.path, d, flush=True)
        if self.path == "/api/config":
            if d.get("brain_url") == "bad":
                return self._json({"detail": "boom"}, 500)
            for k, v in d.items():
                if k == "api_key":
                    STATE["config"]["api_key_set"] = bool(v)
                else:
                    STATE["config"][k] = v
            return self._json({"ok": True})
        if self.path == "/api/robot/move":
            if d.get("value") == "fail":
                return self._json({"ok": False, "error": "Robot motion subsystem not ready"})
            STATE["motion"]["last"] = f"{d.get('kind')}: {d.get('value')}"
            return self._json({"ok": True, "result": f"{d.get('kind')} {d.get('value') or 'random'}"})
        self._json({"ok": True})

    def log_message(self, *a):
        pass


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
