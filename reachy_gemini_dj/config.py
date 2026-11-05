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
    model: str = "gemini-2.0-flash-exp"
    voice: str = "Puck"
    extra_instructions: str = ""
    music_volume: float = 0.70
    duck_level: float = 0.20
    voice_gain: float = 1.00
    echo_guard: float = 1.50
    web_search: bool = False
    motion_enabled: bool = True
    expressive: bool = True

    def save(self) -> None:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = CONFIG_FILE.with_suffix(".tmp")
        data = asdict(self)
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        os.replace(tmp, CONFIG_FILE)

    @classmethod
    def load(cls) -> Config:
        cfg = cls()
        env_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if env_key:
            cfg.api_key = env_key.strip()

        if CONFIG_FILE.exists():
            try:
                data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
                for k, v in data.items():
                    if hasattr(cfg, k):
                        setattr(cfg, k, v)
            except Exception as e:
                log.warning("Could not read config.json (%s)", e)
        return cfg
