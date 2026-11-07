"""Saved favorite songs (~/.reachy_gemini_dj/favorites.json). Atomic writes, max 200, deduped by video id / title."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

from .config import HOME

log = logging.getLogger(__name__)

FAV_FILE = HOME / "favorites.json"   # module-level so tests can point it elsewhere
MAX_FAVORITES = 200
_lock = threading.Lock()


def _path() -> Path:
    return Path(FAV_FILE)


def load() -> list[dict[str, Any]]:
    try:
        data = json.loads(_path().read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except Exception as e:
        log.warning("Could not read favorites (%s)", e)
        return []
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict) and (d.get("id") or d.get("title"))]


def _save(items: list[dict[str, Any]]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(items, indent=2), encoding="utf-8")
    os.replace(tmp, p)


def _same(a: dict, b: dict) -> bool:
    if a.get("id") and a.get("id") == b.get("id"):
        return True
    ta, tb = (a.get("title") or "").strip().lower(), (b.get("title") or "").strip().lower()
    return bool(ta) and ta == tb


def add(title: str, artist: str = "", video_id: str = "", url: str = "") -> tuple[bool, dict[str, Any]]:
    """Returns (added, entry). added is False when it was already a favorite."""
    entry = {"id": video_id or "", "title": title, "artist": artist or "", "url": url or "", "saved_at": int(time.time())}
    with _lock:
        items = load()
        for it in items:
            if _same(it, entry):
                return False, it
        items.append(entry)
        if len(items) > MAX_FAVORITES:
            items = items[-MAX_FAVORITES:]
        _save(items)
    return True, entry


def remove(fav_id: str) -> bool:
    """Remove by video id (or, for entries without one, by exact title)."""
    with _lock:
        items = load()
        keep = [i for i in items if (i.get("id") or i.get("title")) != fav_id]
        if len(keep) == len(items):
            return False
        _save(keep)
    return True
