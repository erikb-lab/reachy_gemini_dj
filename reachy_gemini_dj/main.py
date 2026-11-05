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
from fastapi.responses import FileResponse, HTMLResponse
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
    dance_enabled: bool | None = None


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

        # Subsystems initialized at startup
        self.music = MusicPlayer(volume=self.config.music_volume)
        self.motion: Motion | None = None
        self.audio_pipe: AudioPipeline | None = None
        self.gemini_session = GeminiSession(
            config=self.config,
            music=self.music,
            motion=None,
            google_account=self.google_account,
            on_event=self.record_event,
            on_voice_chunk=lambda chunk: self.audio_pipe.play_voice(chunk) if self.audio_pipe else None,
            on_barge_in=lambda: self.audio_pipe.clear_voice() if self.audio_pipe else None,
            on_heard=self._set_heard,
            on_said=self._set_said,
        )

        # Register routes immediately so that when wrapped_run() launches FastAPI, routes exist!
        self._setup_routes()

    def _set_heard(self, text: str) -> None:
        self.heard_text = (self.heard_text + " " + text).strip()[-300:]

    def _set_said(self, text: str) -> None:
        self.said_text = (self.said_text + " " + text).strip()[-300:]

    def record_event(self, kind: str, text: str) -> None:
        t = datetime.datetime.now().strftime("%H:%M:%S")
        self.events.appendleft({"t": t, "kind": kind, "text": text})

    def _setup_routes(self) -> None:
        app = self.settings_app

        @app.get("/")
        def serve_index():
            index_file = STATIC_DIR / "index.html"
            if index_file.exists():
                return FileResponse(index_file)
            return HTMLResponse("<h2>Reachy Gemini DJ starting...</h2><p>Please wait a moment and refresh.</p>")

        if STATIC_DIR.exists():
            app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

        @app.get("/api/state")
        def get_state():
            g_state = self.gemini_session.state if self.gemini_session else "starting"
            g_err = self.gemini_session.last_error if self.gemini_session else ""
            m_status = (
                self.motion.status()
                if self.motion
                else {
                    "enabled": self.config.motion_enabled,
                    "busy": False,
                    "last": "",
                    "emotions_ready": False,
                    "dances_ready": False,
                    "error": "",
                }
            )

            return {
                "version": "0.2.2",
                "gemini": g_state,
                "model": self.config.model,
                "error": g_err,
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
                    "dance_enabled": self.config.dance_enabled,
                },
                "music": self.music.get_status(),
                "google": self.google_account.public(),
                "motion": m_status,
                "events": list(self.events),
            }

        @app.post("/api/config")
        def update_config(data: ConfigUpdate):
            d = data.model_dump(exclude_unset=True)
            for k, v in d.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.config.save()
            self.music.set_volume(self.config.music_volume)
            if "api_key" in d and self.gemini_session:
                self.gemini_session.init_client()
            return {"ok": True}

        @app.post("/api/mute")
        def set_mute(data: MuteUpdate):
            self.muted = data.muted
            if self.audio_pipe:
                self.audio_pipe.set_mute(self.muted)
            return {"ok": True, "muted": self.muted}

        @app.post("/api/new_conversation")
        def new_conversation():
            self.heard_text = ""
            self.said_text = ""
            if self.gemini_session:
                self.gemini_session.init_client()
            return {"ok": True}

        # Music endpoints
        @app.post("/api/music/pause")
        def music_pause():
            return self.music.pause()

        @app.post("/api/music/resume")
        def music_resume():
            return self.music.resume()

        @app.post("/api/music/skip")
        def music_skip():
            return self.music.skip()

        @app.post("/api/music/stop")
        def music_stop():
            return self.music.stop()

        # Robot endpoints
        @app.get("/api/robot/moves")
        def list_moves():
            return {
                "emotions": EMOTION_NAMES,
                "dances": self.motion.dance_names() if self.motion else [],
                "status": self.motion.status() if self.motion else {},
            }

        @app.post("/api/robot/move")
        def trigger_move(req: MoveRequest):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                if req.kind == "head":
                    res = self.motion.move_head(req.value)
                elif req.kind == "emotion":
                    res = self.motion.emotion(req.value)
                elif req.kind == "dance":
                    res = self.motion.dance(req.value)
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

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        log.info("Starting Reachy Gemini DJ v0.2.2...")
        self.record_event("info", "Reachy Gemini DJ starting")

        # Initialize Motion subsystem connected to live robot
        self.motion = Motion(
            mini=reachy_mini,
            stop=stop_event,
            enabled=lambda: self.config.motion_enabled,
            dance_enabled=lambda: self.config.dance_enabled,
        )
        self.gemini_session.motion = self.motion

        self.audio_pipe = AudioPipeline(
            mini=reachy_mini,
            stop=stop_event,
            config=self.config,
            music=self.music,
            on_level=lambda lvl: setattr(self, "mic_level", lvl),
            on_pcm_chunk=self.gemini_session.push_mic,
        )

        # Start background pipelines
        self.audio_pipe.start()
        self.gemini_session.start()

        # Run expressive subtle reactions while idle
        while not stop_event.is_set():
            if self.config.expressive and self.config.motion_enabled and self.motion and not self.motion.busy and not self.music.is_playing:
                time.sleep(12.0)
                if not self.motion.busy and not stop_event.is_set():
                    self.motion.move_head("front")
            else:
                time.sleep(1.0)

        log.info("Stopping Reachy Gemini DJ...")
        if self.gemini_session:
            self.gemini_session.stop()
        if self.audio_pipe:
            self.audio_pipe.stop()
        self.music.stop()


if __name__ == "__main__":
    app = ReachyGeminiDJ()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
