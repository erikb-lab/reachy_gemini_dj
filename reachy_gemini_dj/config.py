"""Configuration management for Reachy Gemini DJ."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

log = logging.getLogger(__name__)

HOME = Path.home() / ".reachy_gemini_dj"
CONFIG_FILE = HOME / "config.json"


@dataclass
class Config:
    api_key: str = ""
    model: str = "gemini-2.5-flash-native-audio-latest"
    voice: str = "Puck"
    extra_instructions: str = ""
    music_volume: float = 2.00  # linear gain 0.0-2.5 (1.0 = unity, 2.0 = 200%); see gain.py
    music_autoplay: bool = True   # keep the music going with related tracks when the queue runs dry
    duck_level: float = 0.20  # fraction 0.0-1.0 of music_volume while someone talks
    voice_gain: float = 1.50
    hw_speaker_percent: int = 100   # speaker chip's own level, set at start (the robot ships at 72% = -17 dB); 0 = leave alone
    echo_guard: float = 1.50
    web_search: bool = False
    motion_enabled: bool = True
    expressive: bool = True
    dance_enabled: bool = False
    # Voice backend: "auto" = local brain first, Gemini Live if it fails; "local"; "gemini"
    backend: str = "auto"
    brain_url: str = "ws://127.0.0.1:8765"
    allow_gemini_assist: bool = True
    # FRC (FIRST Robotics): team/event can be filled by scanning a Nexus QR code
    frc_team: str = ""
    frc_event: str = ""
    nexus_api_key: str = ""
    tba_api_key: str = ""
    nexus_link: str = ""          # private team link from the Nexus QR; never shown or sent anywhere
    frc_alerts: bool = True
    # How the mic is used: "always" = always listening; "push_to_talk" = only while talk is held (loud events)
    listen_mode: str = "always"
    greet_on_start: bool = True   # say "I'm ready." once when the voice backend first becomes ready after app start
    imu_reactions: bool = True    # speak / react when picked up, set down, bumped, shaken or tipped over (imu.py)
    timezone: str = ""            # IANA zone for alarms, e.g. "America/Chicago"; filled from the controller page

    @property
    def effective_model(self) -> str:
        m = (self.model or "").strip()
        low = m.lower()
        if not m or "dialog" in low or "flash native" in low or "2.0-flash" in low:
            return "gemini-2.5-flash-native-audio-latest"
        if not m.startswith("models/") and not m.startswith("gemini-"):
            return "gemini-2.5-flash-native-audio-latest"
        return m

    def save(self) -> None:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        data = asdict(self)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, 0o600)          # holds API keys and the private Nexus link: owner-only
        os.replace(tmp, CONFIG_FILE)

    @classmethod
    def load(cls) -> Config:
        cfg = cls()
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key:
            cfg.api_key = env_key.strip()
        if os.environ.get("BRAIN_URL"):
            cfg.brain_url = os.environ["BRAIN_URL"].strip()

        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception as e:
                log.warning("Could not read config.json (%s)", e)
        return cfg
