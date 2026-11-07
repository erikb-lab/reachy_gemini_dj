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

from .audio import AudioPipeline, set_hw_speaker_level
from .backends import VoiceBackend
from .config import Config
from .frc import FRC
from .timers import Timers
from .tool_schemas import robot_tool_specs
from .gemini import GeminiSession, VOICES
from .google import GoogleAccount
from .motion import Motion, EMOTION_NAMES
from .gain import clamp_duck, clamp_volume
from .music import MusicPlayer
from .remote_session import RemoteSession

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
    hw_speaker_percent: int | None = None
    echo_guard: float | None = None
    web_search: bool | None = None
    motion_enabled: bool | None = None
    expressive: bool | None = None
    dance_enabled: bool | None = None
    backend: str | None = None
    brain_url: str | None = None
    allow_gemini_assist: bool | None = None
    frc_team: str | None = None
    frc_event: str | None = None
    nexus_api_key: str | None = None
    tba_api_key: str | None = None
    frc_alerts: bool | None = None


FRC_KEYS = {"nexus_api_key", "tba_api_key"}
VOICE_KEYS = {"api_key", "backend", "brain_url", "allow_gemini_assist", "extra_instructions", "voice", "model", "web_search"}


class MuteUpdate(BaseModel):
    muted: bool


class TimerCancel(BaseModel):
    id: int


class MoveRequest(BaseModel):
    kind: str
    value: str


class PoseName(BaseModel):
    name: str


class LearnToggle(BaseModel):
    on: bool


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
        self._mini: ReachyMini | None = None
        self.local_session = RemoteSession(
            config=self.config,
            execute_tool=self.gemini_session.execute_tool,
            on_event=self.record_event,
            on_voice_chunk=lambda chunk: self.audio_pipe.play_voice(chunk) if self.audio_pipe else None,
            on_barge_in=lambda: self.audio_pipe.clear_voice() if self.audio_pipe else None,
            on_heard=self._set_heard_turn,
            on_said=self._set_said,
            is_speaking=lambda: bool(self.audio_pipe and self.audio_pipe._is_robot_speaking),
            tool_specs=self._tool_specs,
            get_frame=self._get_frame,
            seconds_since_user_spoke=lambda: (time.monotonic() - (self.audio_pipe._last_user_speech - 0.7))
            if self.audio_pipe else 1e9,
        )
        # Local brain first, Gemini Live as fallback (see backends.py); both share execute_tool.
        self.voice = VoiceBackend(
            self.config, self.local_session, self.gemini_session, self.record_event,
            is_idle=lambda: not (self.audio_pipe and (self.audio_pipe._is_robot_speaking or self.audio_pipe._user_speaking)),
        )

        # FRC: Nexus QR scan, live queuing alerts, Blue Alliance rankings (tools run on the robot; keys stay here)
        self.frc = FRC(self.config, get_frame=self._get_frame, say=self.voice.say, on_event=self.record_event)
        self.gemini_session.frc = self.frc

        # Timers / reminders / alarms: persisted, rung through the voice backend; user speech dismisses a ringing one
        self.timers = Timers(
            say=self.voice.say, on_event=self.record_event, music=self.music,
            play_pcm=lambda pcm: self.audio_pipe.play_voice(pcm) if self.audio_pipe else None,
            user_speech_age=lambda: (time.monotonic() - (self.audio_pipe._last_user_speech - 0.7))
            if self.audio_pipe else 1e9,
        )
        self.gemini_session.timers = self.timers

        # Register routes immediately so that when wrapped_run() launches FastAPI, routes exist!
        self._setup_routes()

    def _get_frame(self):
        media = getattr(self._mini, "media", None)
        fn = getattr(media, "get_frame", None)
        return fn() if callable(fn) else None

    def _set_heard(self, text: str) -> None:
        self.heard_text = (self.heard_text + " " + text).strip()[-300:]

    def _set_heard_turn(self, text: str) -> None:
        """Local brain: 'heard' is one complete utterance and starts a new turn (Gemini streams fragments instead)."""
        self.heard_text = text.strip()[-300:]
        self.said_text = ""

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
            g_state = self.voice.state
            g_err = self.voice.last_error
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
                "version": "0.5.1",
                "gemini": g_state,
                "model": self.config.model if self.voice.active_name == "gemini" else "local brain",
                "error": g_err,
                "muted": self.muted,
                "mic_level": self.mic_level,
                "heard": self.heard_text,
                "said": self.said_text,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "voices": VOICES,
                "active_backend": self.voice.active_name,
                "brain": self.voice.status(),
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
                    "backend": self.config.backend,
                    "brain_url": self.config.brain_url,
                    "allow_gemini_assist": self.config.allow_gemini_assist,
                    "frc_team": self.config.frc_team,
                    "frc_event": self.config.frc_event,
                    "frc_alerts": self.config.frc_alerts,
                    "nexus_key_set": bool(self.config.nexus_api_key),
                    "tba_key_set": bool(self.config.tba_api_key),
                    "nexus_linked": bool(self.config.nexus_link),
                },
                "frc": dict(self.frc.status),
                "timers": self.timers.status(),
                "music": self.music.get_status(),
                "google": self.google_account.public(),
                "motion": m_status,
                "events": list(self.events),
            }

        @app.post("/api/config")
        def update_config(data: ConfigUpdate):
            d = data.model_dump(exclude_unset=True)
            if d.get("brain_url") and not str(d["brain_url"]).strip().lower().startswith(("ws://", "wss://")):
                return {"ok": False, "error": "Brain URL must start with ws:// or wss://"}
            if d.get("music_volume") is not None:
                d["music_volume"] = clamp_volume(d["music_volume"])
            if d.get("duck_level") is not None:
                d["duck_level"] = clamp_duck(d["duck_level"])
            for k, v in d.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.config.save()
            self.music.set_volume(self.config.music_volume)
            if d.get("hw_speaker_percent") is not None:
                set_hw_speaker_level(self.config.hw_speaker_percent)
            if (VOICE_KEYS | FRC_KEYS) & d.keys():
                self.voice.init_client()          # the local brain's tool list depends on the FRC keys
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
            self.voice.init_client()
            return {"ok": True}

        # Music endpoints
        class MusicPlayRequest(BaseModel):
            query: str

        @app.post("/api/music/play")
        def music_play(req: MusicPlayRequest):
            return self.music.play(req.query)

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

        @app.post("/api/timers/cancel")
        def timers_cancel(req: TimerCancel):
            res = self.timers.cancel_id(req.id)
            return {"ok": "error" not in res, **res}

        @app.post("/api/timers/stop")
        def timers_stop():
            return {"ok": True, **self.timers.stop_alarm()}

        @app.post("/api/frc/scan")
        def frc_scan():
            res = self.frc.scan_qr()               # looks through the camera for up to ~6 s
            return {"ok": "error" not in res, **res}

        # Taught poses: relax motors, move the robot by hand, save (written to disk right away)
        @app.get("/api/poses")
        def list_poses():
            if not self.motion:
                return {"ok": False, "learning": False, "poses": [], "error": "Robot motion subsystem not ready"}
            return {"ok": True, "learning": self.motion.learning, "poses": self.motion.poses.listing()}

        @app.post("/api/poses/learn")
        def learn_mode(req: LearnToggle):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                res = self.motion.learn_start() if req.on else self.motion.learn_stop()
                self.record_event("pose", res)
                return {"ok": True, "learning": self.motion.learning, "result": res}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/poses/save")
        def save_pose(req: PoseName):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                name = self.motion.save_pose(req.name)
                self.record_event("pose", f"Saved pose '{name}'")
                return {"ok": True, "name": name, "poses": self.motion.poses.listing()}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/poses/play")
        def play_pose(req: PoseName):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                return {"ok": True, "name": self.motion.goto_pose(req.name)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/poses/delete")
        def delete_pose(req: PoseName):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                ok = self.motion.poses.delete(req.name)
                return {"ok": ok, "poses": self.motion.poses.listing(), **({} if ok else {"error": "No such pose"})}
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
                self.voice.init_client()          # now offer the Google tools to the local brain
                return {"ok": True, "public": res}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/google/disconnect")
        def google_disconnect():
            self.google_account.disconnect()
            self.voice.init_client()
            return {"ok": True}

    def _tool_specs(self) -> list[dict]:
        """Tools offered to the local brain: only those that can work (each one costs prompt tokens to read)."""
        return robot_tool_specs(google_services=set(self.google_account.granted()) if self.google_account.connected else set(),
                                frc_enabled=bool(self.config.nexus_api_key or self.config.tba_api_key))

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        log.info("Starting Reachy Gemini DJ v0.5.1...")
        self.record_event("info", "Reachy Gemini DJ starting")

        # Initialize Motion subsystem connected to live robot
        self.motion = Motion(
            mini=reachy_mini,
            stop=stop_event,
            enabled=lambda: self.config.motion_enabled,
            dance_enabled=lambda: self.config.dance_enabled,
        )
        self.voice.motion = self.motion
        self._mini = reachy_mini

        self.audio_pipe = AudioPipeline(
            mini=reachy_mini,
            stop=stop_event,
            config=self.config,
            music=self.music,
            on_level=lambda lvl: setattr(self, "mic_level", lvl),
            on_pcm_chunk=self.voice.push_mic,
        )

        # Start background pipelines
        self.audio_pipe.start()
        self.voice.start()
        self.frc.start()
        self.timers.motion = self.motion
        self.timers.start()

        # Run expressive subtle reactions while idle
        while not stop_event.is_set():
            if self.config.expressive and self.config.motion_enabled and self.motion and not self.motion.busy and not self.music.is_playing:
                time.sleep(12.0)
                if not self.motion.busy and not stop_event.is_set():
                    try:
                        self.motion.move_head("front")
                    except Exception as e:
                        log.debug("Expressive idle move note: %s", e)
            else:
                time.sleep(1.0)

        log.info("Stopping Reachy Gemini DJ...")
        if self.motion and self.motion.learning:
            try:
                self.motion.learn_stop()           # never leave the motors relaxed
            except Exception:
                pass
        self.frc.stop()
        self.timers.stop()
        self.voice.stop()
        if self.audio_pipe:
            self.audio_pipe.stop()
        self.music.stop()


if __name__ == "__main__":
    app = ReachyGeminiDJ()
    try:
        app.wrapped_run()
    except KeyboardInterrupt:
        app.stop()
