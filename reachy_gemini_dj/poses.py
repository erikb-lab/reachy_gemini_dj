"""Poses taught by hand: relax the motors, move the robot, save. Saved straight to ~/.reachy_gemini_dj/poses.json."""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from pathlib import Path

from .config import HOME

log = logging.getLogger(__name__)

POSES_FILE = HOME / "poses.json"
MAX_POSES = 100


def clean_name(name: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "")).strip()[:40]
    if not n:
        raise ValueError("Give the pose a name.")
    return n


class PoseStore:
    def __init__(self, path: Path = POSES_FILE):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._poses: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._poses = {k: v for k, v in data.items() if isinstance(v, dict) and "head" in v}
        except FileNotFoundError:
            pass
        except Exception as e:
            log.warning("Could not read %s (%s); starting with no poses", self.path, e)

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._poses, indent=1), encoding="utf-8")
        os.replace(tmp, self.path)              # atomic: a crash never leaves a half-written file

    def find(self, name: str) -> tuple[str, dict] | None:
        key = clean_name(name).lower()
        with self._lock:
            for k, v in self._poses.items():
                if k.lower() == key:
                    return k, v
            for k, v in self._poses.items():     # forgiving match for voice: "wave" finds "big wave"
                if key in k.lower():
                    return k, v
        return None

    def save(self, name: str, pose: dict) -> str:
        name = clean_name(name)
        with self._lock:
            existing = next((k for k in self._poses if k.lower() == name.lower()), None)
            if existing is None and len(self._poses) >= MAX_POSES:
                raise ValueError(f"Too many saved poses (max {MAX_POSES}); delete some first.")
            self._poses.pop(existing, None) if existing else None
            self._poses[name] = {**pose, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
            self._write()
        return name

    def delete(self, name: str) -> bool:
        found = self.find(name)
        if not found:
            return False
        with self._lock:
            self._poses.pop(found[0], None)
            self._write()
        return True

    def names(self) -> list[str]:
        with self._lock:
            return sorted(self._poses, key=str.lower)

    def listing(self) -> list[dict]:
        with self._lock:
            return [{"name": k, "saved_at": v.get("saved_at", "")} for k, v in sorted(self._poses.items(), key=lambda kv: kv[0].lower())]
