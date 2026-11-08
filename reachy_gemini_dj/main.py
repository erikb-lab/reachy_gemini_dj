"""Main entry point for Reachy Gemini DJ app.

Extends Pollen's ReachyMiniApp, serving the FastAPI web UI on port 8042 and orchestrating
audio capture, Gemini live voice streaming, music playback, and robot gestures.
"""
from __future__ import annotations

import collections
import datetime
import logging
import re
import shutil
import threading
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from reachy_mini import ReachyMini, ReachyMiniApp

from .audio import AudioPipeline, LISTEN_MODES, set_hw_speaker_level
from .backends import VoiceBackend
from .config import Config
from .frc import FRC
from .timers import Timers, set_timezone, _dt as local_dt
from .tool_schemas import robot_tool_specs
from .gemini import GeminiSession, VOICES
from .google import GoogleAccount
from .motion import Motion, EMOTION_NAMES
from .motor_health import MotorHealth
from .panorama import PanoramaJob
from .gain import clamp_duck, clamp_volume
from .memory import MAX_CHARS, MAX_FACTS, Memory
from .music import MusicPlayer
from .remote_session import RemoteSession
from .imu import ImuReactions, ImuWatcher
from .qol import ConversationLog, Greeter, QuietMode, brain_status

log = logging.getLogger("reachy_gemini_dj")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

STATIC_DIR = Path(__file__).parent / "static"


class ConfigUpdate(BaseModel):
    api_key: str | None = None
    extra_instructions: str | None = None
    model: str | None = None
    voice: str | None = None
    music_volume: float | None = None
    music_autoplay: bool | None = None
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
    listen_mode: str | None = None          # "always" | "push_to_talk"
    timezone: str | None = None             # IANA name, sent by the controller page
    greet_on_start: bool | None = None
    imu_reactions: bool | None = None
    music_eq_bass: float | None = None
    music_eq_mid: float | None = None
    music_eq_treble: float | None = None
    gamepad_enabled: bool | None = None
    gamepad_speed: float | None = None


EQ_KEYS = ("music_eq_bass", "music_eq_mid", "music_eq_treble")
EQ_RANGE_DB = 12.0
FRC_KEYS = {"nexus_api_key", "tba_api_key"}
VOICE_KEYS = {"api_key", "backend", "brain_url", "allow_gemini_assist", "extra_instructions", "voice", "model", "web_search"}


class MemoryAdd(BaseModel):
    text: str


class MemoryDelete(BaseModel):
    id: str


class PttRequest(BaseModel):
    down: bool


class MusicPlayRequest(BaseModel):
    query: str


class MusicQueueRequest(BaseModel):
    action: str
    which: str = ""
    id: int | None = None


class MusicFavoriteRequest(BaseModel):
    action: str
    id: str = ""


class MusicSeekRequest(BaseModel):
    to: str


class MuteUpdate(BaseModel):
    muted: bool


class QuietRequest(BaseModel):
    minutes: int = 30


class TimerCancel(BaseModel):
    id: int


class MoveRequest(BaseModel):
    kind: str
    value: str


class PoseName(BaseModel):
    name: str


class LearnToggle(BaseModel):
    on: bool


class DriveToggle(BaseModel):
    on: bool


class DriveTarget(BaseModel):
    """Gamepad drive target: mm for x/y/z, degrees for the rest; missing values are left as they are."""
    x: float | None = None
    y: float | None = None
    z: float | None = None
    roll: float | None = None
    pitch: float | None = None
    yaw: float | None = None
    body: float | None = None
    ant_l: float | None = None
    ant_r: float | None = None


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
        self.talk_pressed_hooks: list = []

        # Subsystems initialized at startup
        self.music = MusicPlayer(volume=self.config.music_volume, autoplay=self.config.music_autoplay)
        self.music.set_eq(self.config.music_eq_bass, self.config.music_eq_mid, self.config.music_eq_treble)
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
        self.motor_health = MotorHealth()
        self.local_session = RemoteSession(
            config=self.config,
            execute_tool=self.gemini_session.execute_tool,
            on_event=self.record_event,
            on_voice_chunk=lambda chunk: self.audio_pipe.play_voice(chunk) if self.audio_pipe else None,
            on_barge_in=lambda: self.audio_pipe.clear_voice() if self.audio_pipe else None,
            on_heard=self._set_heard_turn,
            on_said=self._set_said,
            on_state=self._on_brain_state,
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

        # Long-term memory ("remember our pit is A3"): tools use it via execute_tool, both brains get the facts
        self.memory = Memory()
        self.gemini_session.memory = self.memory
        self.local_session.get_memory = self.memory.texts
        self._memory_sync_lock = threading.Lock()
        self._memory_sync_pending = False

        # FRC: Nexus QR scan, live queuing alerts, Blue Alliance rankings (tools run on the robot; keys stay here)
        self.frc = FRC(self.config, get_frame=self._get_frame, say=self.voice.say, on_event=self.record_event)
        self.gemini_session.frc = self.frc

        # Timers / reminders / alarms: persisted, rung through the voice backend; user speech dismisses a ringing one
        self.timers = Timers(
            say=self.voice.say, on_event=self.record_event, music=self.music,
            play_pcm=lambda pcm: self.audio_pipe.play_voice(pcm) if self.audio_pipe else None,
            # In push-to-talk mode loud rooms are the norm, so mic noise never dismisses; the talk button does.
            user_speech_age=lambda: (time.monotonic() - (self.audio_pipe._last_user_speech - 0.7))
            if self.audio_pipe and self.config.listen_mode != "push_to_talk" else 1e9,
        )
        self.gemini_session.timers = self.timers
        self.talk_pressed_hooks.append(self.timers.dismiss)
        set_timezone(self.config.timezone)

        # Quality of life (qol.py): conversation log, quiet mode, "I'm ready" greeting
        self.convo = ConversationLog(backend=lambda: self.voice.active_name)
        self.quiet = QuietMode(set_muted=self._set_muted, say=self.voice.say, on_event=self.record_event)
        self.gemini_session.quiet = self.quiet
        self.talk_pressed_hooks.append(self.quiet.on_talk)
        self.greeter = Greeter(say=self.voice.say, enabled=lambda: self.config.greet_on_start, is_quiet=lambda: self.quiet.active)
        self.local_session.on_ready = lambda: self.greeter.ready("local")
        self.gemini_session.on_connected = lambda: self.greeter.ready("gemini")
        self.local_session.on_turn_done = self.convo.turn_done
        self.gemini_session.on_turn_done = self.convo.turn_done

        # IMU (imu.py): events (picked up / bumped / shaken / tipped over) -> short spoken reactions; robot_sensors tool
        self.imu = ImuWatcher(
            read_imu=self._read_imu, suppress=self._imu_suppressed,
            no_bump=lambda: self.music.is_playing or bool(self.audio_pipe and self.audio_pipe._is_robot_speaking),
            on_event=lambda kind, info: self.imu_reactions.on_event(kind, info),
        )
        self.imu_reactions = ImuReactions(
            say=self.voice.say, emotion=lambda name: self.motion.emotion(name) if self.motion else None,
            enabled=lambda: self.config.imu_reactions, is_idle=self.voice.is_idle,
            is_quiet=lambda: self.quiet.active,
            motion_free=lambda: bool(self.motion and self.config.motion_enabled and not self.motion.moving()),
        )
        self.gemini_session.imu = self.imu

        # 360° photos (panorama.py): the robot turns through a grid of poses and stitches a photo sphere on the robot
        self.panorama = PanoramaJob(get_mini=lambda: self._mini, on_event=self.record_event,
                                    motion_enabled=lambda: self.config.motion_enabled)
        self.gemini_session.panorama = self.panorama

        # Register routes immediately so that when wrapped_run() launches FastAPI, routes exist!
        self._setup_routes()

    # Callbacks run whenever talk is pressed (push-to-talk button, space bar, /api/ptt). Other features append here,
    # e.g. timers: `app.talk_pressed_hooks.append(timers.dismiss_alarm)`. Each is called with no arguments.
    def _on_talk_pressed(self) -> None:
        for fn in list(self.talk_pressed_hooks):
            try:
                fn()
            except Exception:
                log.exception("talk-pressed hook failed")

    def _on_brain_state(self, state: str, detail: str = "") -> None:
        """Local brain reported its state: tilt the head while it thinks (never for Gemini Live)."""
        m = self.motion
        if m is not None:
            m.thinking(state == "thinking")

    def _get_frame(self):
        media = getattr(self._mini, "media", None)
        fn = getattr(media, "get_frame", None)
        return fn() if callable(fn) else None

    def _read_imu(self):
        """Latest cached IMU dict from the SDK (50 Hz cache, cheap), or None (Lite / simulator / stale)."""
        return getattr(self._mini, "imu", None)

    def _imu_suppressed(self) -> bool:
        """Ignore the IMU while Reachy moves itself, is limp for hand posing, or is in quiet mode."""
        m = self.motion
        return bool(self.quiet.active or (m is not None and m.moving()) or self.panorama.busy())

    def _set_muted(self, muted: bool) -> None:
        self.muted = bool(muted)
        if self.audio_pipe:
            self.audio_pipe.set_mute(self.muted)

    def _set_heard(self, text: str) -> None:
        self.convo.heard_fragment(text)
        self.heard_text = (self.heard_text + " " + text).strip()[-300:]

    def _set_heard_turn(self, text: str) -> None:
        """Local brain: 'heard' is one complete utterance and starts a new turn (Gemini streams fragments instead)."""
        self.convo.heard_utterance(text)
        self.heard_text = text.strip()[-300:]
        self.said_text = ""

    def _set_said(self, text: str) -> None:
        self.convo.said(text)
        self.said_text = (self.said_text + " " + text).strip()[-300:]

    def record_event(self, kind: str, text: str) -> None:
        t = local_dt(time.time()).strftime("%H:%M:%S")
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
                "version": "0.8.55",
                "gemini": g_state,
                "model": self.config.model if self.voice.active_name == "gemini" else "local brain",
                "error": g_err,
                "muted": self.muted,
                "mic_level": self.mic_level,
                "listen_mode": self.audio_pipe.listen_mode if self.audio_pipe else self.config.listen_mode,
                "ptt_active": bool(self.audio_pipe and self.audio_pipe.ptt_active),
                "heard": self.heard_text,
                "said": self.said_text,
                "ffmpeg": bool(shutil.which("ffmpeg")),
                "voices": VOICES,
                "active_backend": self.voice.active_name,
                "brain": brain_status(self.voice.status(), self.local_session.state, self.convo.last_latency("local")),
                "conversation": self.convo.listing(),
                "quiet": self.quiet.status(),
                "imu": self.imu.status(),
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
                    "timezone": self.config.timezone,
                    "greet_on_start": self.config.greet_on_start,
                    "imu_reactions": self.config.imu_reactions,
                    "music_eq_bass": self.config.music_eq_bass,
                    "music_eq_mid": self.config.music_eq_mid,
                    "music_eq_treble": self.config.music_eq_treble,
                    "gamepad_enabled": self.config.gamepad_enabled,
                    "gamepad_speed": self.config.gamepad_speed,
                    "backend": self.config.backend,
                    "brain_url": self.config.brain_url,
                    "allow_gemini_assist": self.config.allow_gemini_assist,
                    "frc_team": self.config.frc_team,
                    "frc_event": self.config.frc_event,
                    "frc_alerts": self.config.frc_alerts,
                    "listen_mode": self.config.listen_mode,
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
            if d.get("frc_event") is not None:
                d["frc_event"] = re.sub(r"\s+", "", str(d["frc_event"])).lower()
                if d["frc_event"] and not re.fullmatch(r"[a-z0-9]{4,}", d["frc_event"]):
                    return {"ok": False, "error": "Event code looks like 2026mndu (letters and digits only)"}
            if d.get("frc_team") is not None:
                d["frc_team"] = re.sub(r"\D", "", str(d["frc_team"]))
            if d.get("brain_url") is not None and str(d["brain_url"]).strip() != (self.config.brain_url or "").strip():
                # memory facts and your notes go to whichever brain this points at: make every change visible
                self.record_event("security", f"Brain URL changed to {str(d['brain_url']).strip()}")
            if d.get("listen_mode") is not None and d["listen_mode"] not in LISTEN_MODES:
                return {"ok": False, "error": "listen_mode must be one of: " + ", ".join(LISTEN_MODES)}
            if d.get("timezone") is not None and not set_timezone(str(d["timezone"]).strip()):
                return {"ok": False, "error": "Unknown time zone"}
            if d.get("music_volume") is not None:
                d["music_volume"] = clamp_volume(d["music_volume"])
            if d.get("duck_level") is not None:
                d["duck_level"] = clamp_duck(d["duck_level"])
            for k in EQ_KEYS:
                if d.get(k) is not None:
                    d[k] = round(max(-EQ_RANGE_DB, min(EQ_RANGE_DB, float(d[k]))), 1)
            if d.get("gamepad_speed") is not None:
                d["gamepad_speed"] = round(max(0.25, min(2.0, float(d["gamepad_speed"]))), 2)
            for k, v in d.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)
            self.config.save()
            self.music.set_volume(self.config.music_volume)
            self.music.autoplay = bool(self.config.music_autoplay)
            if any(k in d for k in EQ_KEYS):
                self.music.set_eq(self.config.music_eq_bass, self.config.music_eq_mid, self.config.music_eq_treble)
            if d.get("gamepad_enabled") is False and self.motion and self.motion.manual:
                self.motion.drive_stop()
            if d.get("hw_speaker_percent") is not None:
                set_hw_speaker_level(self.config.hw_speaker_percent)
            if (VOICE_KEYS | FRC_KEYS) & d.keys():
                self.voice.init_client()          # the local brain's tool list depends on the FRC keys
            return {"ok": True}

        @app.post("/api/mute")
        def set_mute(data: MuteUpdate):
            self._set_muted(data.muted)
            if not data.muted:
                self.quiet.wake(say=False, reason="unmuted")
            return {"ok": True, "muted": self.muted}

        # Quiet mode: POST /api/quiet {"minutes": 30} (1-480) mutes the mic and rests the head until the time is up;
        # POST /api/quiet/wake ends it early. Also the voice tool quiet_mode and pressing talk. See GET /api/state "quiet".
        @app.post("/api/quiet")
        def quiet_start(req: QuietRequest):
            return {"ok": True, **self.quiet.enter(req.minutes), "quiet": self.quiet.status()}

        @app.post("/api/quiet/wake")
        def quiet_wake():
            woke = self.quiet.wake(reason="page")
            return {"ok": True, "woke": woke, "quiet": self.quiet.status()}

        # Push-to-talk. This is the whole API a physical button needs (ESP32 over Wi-Fi/BLE bridge, USB foot pedal or
        # presenter clicker via a tiny script, ...); no auth, so it is only as private as the robot's network:
        #   POST /api/ptt         {"down": true}   press   (start sending the mic; barge-in if Reachy is talking)
        #   POST /api/ptt         {"down": false}  release (~0.9 s of silence is sent so the utterance ends promptly)
        #   POST /api/ptt/toggle  (no body)        tap to start, tap again to stop, for one-button devices
        # Reply: {"ok": true, "listen_mode": "...", "ptt_active": bool}. Holding for more than 30 s auto-releases.
        # In listen_mode "always" a press only does the barge-in / talk-pressed hooks; the mic is untouched.
        # GET /api/state shows "listen_mode" and "ptt_active". Set the mode with POST /api/config {"listen_mode": ...}.
        @app.post("/api/ptt")
        def ptt(req: PttRequest):
            if not self.audio_pipe:
                return {"ok": False, "error": "Audio not ready", "ptt_active": False}
            active = self.audio_pipe.ptt_set(req.down)
            return {"ok": True, "listen_mode": self.audio_pipe.listen_mode, "ptt_active": active}

        @app.post("/api/ptt/toggle")
        def ptt_toggle():
            if not self.audio_pipe:
                return {"ok": False, "error": "Audio not ready", "ptt_active": False}
            active = self.audio_pipe.ptt_toggle()
            return {"ok": True, "listen_mode": self.audio_pipe.listen_mode, "ptt_active": active}

        @app.post("/api/new_conversation")
        def new_conversation():
            self.heard_text = ""
            self.said_text = ""
            self.voice.init_client()
            return {"ok": True}

        # Music endpoints
        @app.post("/api/music/play")
        def music_play(req: MusicPlayRequest):
            return self.music.play(req.query, added_by="controller")

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

        # Queue / favorites / seek (music_queue.js)
        @app.post("/api/music/queue")
        def music_queue(req: MusicQueueRequest):
            if req.action == "remove" and req.id is not None:
                res = self.music.remove_item(req.id)
            elif req.action == "play_next" and req.id is not None:
                res = self.music.move_next(req.id)
            else:
                res = self.music.queue_action(req.action, req.which)
            if req.action.startswith("autoplay"):
                self.config.music_autoplay = self.music.autoplay
                self.config.save()
            return res

        @app.post("/api/music/favorite")
        def music_favorite(req: MusicFavoriteRequest):
            return self.music.favorite(req.action, req.id)

        @app.get("/api/music/favorites")
        def music_favorites():
            from . import favorites
            return {"favorites": favorites.load()}

        @app.post("/api/music/seek")
        def music_seek(req: MusicSeekRequest):
            return self.music.seek(req.to)

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

        # Gamepad mode: the page reads a controller (Gamepad API) and streams a target pose; motion.py moves the
        # motors there at a capped speed and pauses every other move while it is on.
        @app.post("/api/drive")
        def drive_toggle(req: DriveToggle):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            if req.on and not self.config.gamepad_enabled:
                return {"ok": False, "error": "Gamepad mode is off: turn it on in Settings > Experimental."}
            if req.on and not self.config.motion_enabled:
                return {"ok": False, "error": "Robot movement is turned off in Settings."}
            try:
                res = self.motion.drive_start() if req.on else self.motion.drive_stop()
                self.record_event("pose", "Gamepad mode on" if req.on else "Gamepad mode off")
                return {"ok": True, **res}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        @app.post("/api/drive/target")
        def drive_target(req: DriveTarget):
            if not self.motion:
                return {"ok": False, "error": "Robot motion subsystem not ready"}
            try:
                return {"ok": True, **self.motion.drive_set(req.model_dump(exclude_none=True))}
            except Exception as e:
                return {"ok": False, "manual": False, "error": str(e)}

        # Motor health: hardware-error flags per motor, and "clear" = Dynamixel reboot of the faulty motors through
        # the daemon's raw packet websocket (clears an Overload without a power cycle), then torque back on.
        @app.get("/api/motors/health")
        def motors_health():
            try:
                return {"ok": True, **self.motor_health.read()}
            except Exception as e:
                return {"ok": False, "error": f"Could not read the motors: {e}"[:200]}

        @app.post("/api/motors/clear")
        def motors_clear():
            try:
                res = self.motor_health.clear()
            except Exception as e:
                return {"ok": False, "error": f"Could not reach the motors: {e}"[:200]}
            if res["cleared"] and self._mini is not None and not (self.motion and self.motion.learning):
                try:
                    self._mini.enable_motors()     # holds every motor where it is now (no snap)
                except Exception as e:
                    res["torque_error"] = str(e)[:160]
            names = ", ".join(r["label"] for r in res["cleared"].values()) or "none"
            self.record_event("error" if res["faults"] else "pose",
                              f"Motor errors cleared: {names}" if not res["faults"] else
                              "Could not clear: " + ", ".join(res["cleared"][n]["label"] for n in res["faults"]))
            return {"ok": not res["faults"], **res}

        # 360° photos. capture starts the robot turning (~34 photos, then a stitch on the robot); the page polls status.
        # Photos are kept on the robot in ~/.reachy_gemini_dj/panoramas (newest 30), as 2:1 JPEGs with GPano metadata.
        @app.post("/api/panorama/capture")
        def panorama_capture():
            return self.panorama.start("page")

        @app.get("/api/panorama/status")
        def panorama_status():
            return self.panorama.status()

        @app.post("/api/panorama/cancel")
        def panorama_cancel():
            return self.panorama.cancel()

        @app.get("/api/panorama/list")
        def panorama_list():
            return {"items": self.panorama.list()}

        @app.get("/api/panorama/{pid}.jpg")
        def panorama_file(pid: str):
            p = self.panorama.path(pid)
            if not p:
                return JSONResponse({"ok": False, "error": "No such photo"}, status_code=404)
            return FileResponse(p, media_type="image/jpeg", filename=f"reachy-360-{pid}.jpg")

        @app.get("/api/panorama/{pid}/thumb.jpg")
        def panorama_thumb(pid: str):
            p = self.panorama.path(pid, thumb=True)
            if not p:
                return JSONResponse({"ok": False, "error": "No such photo"}, status_code=404)
            return FileResponse(p, media_type="image/jpeg")

        @app.delete("/api/panorama/{pid}")
        def panorama_delete(pid: str):
            return {"ok": self.panorama.delete(pid)}

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

        # Memory: shown and edited on the controller page. Edits made here reach the local brain through a
        # reconnect once the robot is idle (see _memory_changed); voice edits do not, the conversation has them.
        @app.get("/api/memory")
        def get_memory():
            return {"ok": True, "facts": self.memory.listing(), "max": MAX_FACTS, "max_chars": MAX_CHARS}

        @app.post("/api/memory")
        def add_memory(req: MemoryAdd):
            r = self.memory.add(req.text)
            if r["status"] in ("added", "updated"):
                self._memory_changed()
            err = {"empty": "Type something to remember.",
                   "full": f"Memory is full ({MAX_FACTS}); delete something first."}.get(r["status"])
            return {"ok": err is None, "status": r["status"], **({"error": err} if err else {}),
                    "facts": self.memory.listing()}

        @app.post("/api/memory/delete")
        def delete_memory(req: MemoryDelete):
            ok = self.memory.delete_id(req.id)
            if ok:
                self._memory_changed()
            return {"ok": ok, "facts": self.memory.listing(), **({} if ok else {"error": "No such memory"})}

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

    def _memory_changed(self) -> None:
        """The page changed the facts: the local brain only learns them in the hello, so reconnect it - but never
        in the middle of a conversation. Waits (in a background thread, up to 10 minutes) until the robot is idle.
        Gemini Live reads the facts when it next connects; it is not interrupted. Not used for voice edits."""
        with self._memory_sync_lock:
            if self._memory_sync_pending:
                return
            self._memory_sync_pending = True

        def work() -> None:
            def done() -> None:
                with self._memory_sync_lock:
                    self._memory_sync_pending = False       # a later edit schedules a fresh reconnect

            try:
                deadline = time.monotonic() + 600
                while time.monotonic() < deadline:
                    if self.voice.active_name != "local":
                        return
                    if self.voice.is_idle():
                        done()
                        self.voice.init_client()
                        return
                    time.sleep(1.0)
            finally:
                done()

        threading.Thread(target=work, daemon=True, name="memory-sync").start()

    def _tool_specs(self) -> list[dict]:
        """Tools offered to the local brain: only those that can work (each one costs prompt tokens to read)."""
        return robot_tool_specs(google_services=set(self.google_account.granted()) if self.google_account.connected else set(),
                                frc_enabled=bool(self.config.nexus_api_key or self.config.tba_api_key))

    def run(self, reachy_mini: ReachyMini, stop_event: threading.Event) -> None:
        log.info("Starting Reachy Gemini DJ v0.8.55...")
        self.record_event("info", "Reachy Gemini DJ starting")

        # Initialize Motion subsystem connected to live robot
        self.motion = Motion(
            mini=reachy_mini,
            stop=stop_event,
            enabled=lambda: self.config.motion_enabled,
            dance_enabled=lambda: self.config.dance_enabled,
        )
        self.voice.motion = self.motion
        self.panorama.motion = self.motion
        self._mini = reachy_mini

        self.audio_pipe = AudioPipeline(
            mini=reachy_mini,
            stop=stop_event,
            config=self.config,
            music=self.music,
            on_level=lambda lvl: setattr(self, "mic_level", lvl),
            on_pcm_chunk=self.voice.push_mic,
            on_talk_pressed=self._on_talk_pressed,
        )

        # Start background pipelines
        self.audio_pipe.start()
        self.voice.start()
        self.frc.start()
        self.timers.motion = self.motion
        self.timers.start()
        self.quiet.motion = self.motion
        self.quiet.start()
        self.imu.start()

        # Run expressive subtle reactions while idle
        while not stop_event.is_set():
            if self.config.expressive and self.config.motion_enabled and self.motion and not self.motion.busy and not self.music.is_playing and not self.quiet.active:
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
        self.quiet.stop()
        self.imu.stop()
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
