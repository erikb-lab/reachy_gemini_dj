"""Main entry point for Reachy Gemini DJ app.

Extends Pollen's ReachyMiniApp, serving the FastAPI web UI on port 8042 and orchestrating
audio capture, Gemini live voice streaming, music playback, and robot gestures.
"""
from __future__ import annotations

import collections
import datetime
import logging
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp

from .audio import AudioPipeline
from .config import Config
from .gemini import GeminiSession, VOICES
from .google import GoogleAccount
from .motion import Motion, EMOTION_NAMES
from .music import MusicPlayer

log = logging.getLogger("reachy_gemini_dj")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


class ConfigUpdate(BaseModel):
    api_key: str | None = None
    extra_instructions: str | None = None
    model: str | None = None
    voice: str | None = None
    music_volume: float | None = None
    duck_level: float | None = None
    voice_gain: float | None = None
    echo_guard: float | None = None
    web_search: bool | None = None
    motion_enabled: bool | None = None
    expressive: bool | None = None


class MuteUpdate(BaseModel):
    muted: bool


class MoveRequest(BaseModel):
    kind: str
    value: str


class GoogleClientUpdate(BaseModel):
    client_id: str
    client_secret: str


class GoogleStartRequest(BaseModel):
    services: list[str]


class GoogleFinishRequest(BaseModel):
    redirect_url: str


class ReachyGeminiDJ(ReachyMiniApp):
    custom_app_url: str | None = "http://0.0.0.0:8042"

    def __init__(self) -> None:
        super().__init__()
        self.config = Config.load()
        self.google_account = GoogleAccount()
        self.events: collections.deque = collections.deque(maxlen=25)
        self.muted = False
        self.heard_text = ""
        self.said_text = ""
        self.mic_level = 0.0

    def record_event(self, kind: str, text: str) -> None:
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.events.appendleft({"t": t, "kind": kind, "text": text})

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        log.info("Starting Reachy Gemini DJ v0.2.0...")
        self.record_event("info", "Reachy Gemini DJ starting")

        # Initialize Motion & Music subsystems
        motion = Motion(
            mini=reachy_mini,
            stop=stop_event,
            enabled=lambda: self.config.motion_enabled,
        )
        music = MusicPlayer(volume=self.config.music_volume)

        # Initialize Gemini & Audio
        gemini_session = GeminiSession(
            config=self.config,
            music=music,
            motion=motion,
            google_account=self.google_account,
            on_event=self.record_event,
        )
        gemini_session.init_client()

        audio_pipe = AudioPipeline(
            mini=reachy_mini,
            stop=stop_event,
            config=self.config,
            music=music,
            on_level=lambda lvl: setattr(self, "mic_level", lvl),
        )

        # Define REST API routes on self.settings_app
        app = self.settings_app

        @app.get("/api/state")
        def get_state():
            return {
                "version": "0.2.0",
                "gemini": gemini_session.state,
                "model": self.config.model,
                "error": gemini_session.last_error,
                "muted": self.muted,
                "mic_level": self.mic_level,
                "heard": self.heard_text,
                "said": self.said_text,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "voices": VOICES,
                "config": {
                    "api_key_set": bool(self.config.api_key),
                    "model": self.config.model,
                    "voice": self.config.voice,
                    "extra_instructions": self.config.extra_instructions,
                    "music_volume": self.config.music_volume,
                    "duck_level": self.config.duck_level,
                    "voice_gain": self.config.voice_gain,
                    "echo_guard": self.config.echo_guard,
                    "web_search": self.config.web_search,
                    "motion_enabled": self.config.motion_enabled,
                    "expressive": self.config.expressive,
                },
                "music": music.get_status(),
                "google": self.google_account.public(),
                "motion": motion.status(),
                "events": list(self.events),
            }

        @app.post("/api/config")
        def update_config(data: ConfigUpdate):
            d = data.model_dump(exclude_unset=True)
            for k, v in d.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.config.save()
            music.set_volume(self.config.music_volume)
            if "api_key" in d:
                gemini_session.init_client()
            return {"ok": True}

        @app.post("/api/mute")
        def set_mute(data: MuteUpdate):
            self.muted = data.muted
            audio_pipe.set_mute(self.muted)
            return {"ok": True, "muted": self.muted}

        @app.post("/api/new_conversation")
        def new_conversation():
            self.heard_text = ""
            self.said_text = ""
            gemini_session.init_client()
            return {"ok": True}

        # Music endpoints
        @app.post("/api/music/pause")
        def music_pause():
            return music.pause()

        @app.post("/api/music/resume")
        def music_resume():
            return music.resume()

        @app.post("/api/music/skip")
        def music_skip():
            return music.skip()

        @app.post("/api/music/stop")
        def music_stop():
            return music.stop()

        # Robot endpoints
        @app.get("/api/robot/moves")
        def list_moves():
            return {
                "emotions": EMOTION_NAMES,
                "dances": motion.dance_names(),
                "status": motion.status(),
            }

        @app.post("/api/robot/move")
        def trigger_move(req: MoveRequest):
            try:
                if req.kind == "head":
                    res = motion.move_head(req.value)
                elif req.kind == "emotion":
                    res = motion.emotion(req.value)
                elif req.kind == "dance":
                    res = motion.dance(req.value)
                else:
                    return {"ok": False, "error": "Unknown move kind"}
                return {"ok": True, "result": res}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Google endpoints
        @app.post("/api/google/client")
        def set_google_client(req: GoogleClientUpdate):
            try:
                self.google_account.set_client(req.client_id, req.client_secret)
                return {"ok": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/google/start")
        def google_start(req: GoogleStartRequest):
            try:
                url = self.google_account.begin(req.services)
                return {"ok": True, "url": url}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/google/finish")
        def google_finish(req: GoogleFinishRequest):
            try:
                res = self.google_account.finish(req.redirect_url)
                return {"ok": True, "public": res}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/google/disconnect")
        def google_disconnect():
            self.google_account.disconnect()
            return {"ok": True}

        # Static UI mount
        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

            @app.get("/")
            def serve_index():
                return FileResponse(STATIC_DIR / "index.html")

        # Start audio background loop
        audio_pipe.start()

        # Run expressive subtle reactions while idle
        while not stop_event.is_set():
            if self.config.expressive and self.config.motion_enabled and not motion.busy and not music.is_playing:
                time.sleep(12.0)
                if not motion.busy and not stop_event.is_set():
                    motion.move_head("front")
            else:
                time.sleep(1.0)

        log.info("Stopping Reachy Gemini DJ...")
        audio_pipe.stop()
        music.stop()


if __name__ == "__main__":
    app = ReachyGeminiDJ()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
