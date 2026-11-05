"""Built-in robot movement only: head moves (goto_target), Pollen's emotion library and dance library.

Nothing here invents motion. Emotions and dances are the recorded moves Pollen Robotics publishes on
Hugging Face (small motion datasets, cached after the first fetch; this is not music). Moves run on one
worker thread so Gemini never waits for them, and a newer request replaces one that is still waiting.
"""
from __future__ import annotations

import collections
import logging
import queue
import random
import re
import threading
import time

log = logging.getLogger(__name__)

EMOTIONS_REPO = "pollen-robotics/reachy-mini-emotions-library"
DANCES_REPO = "pollen-robotics/reachy-mini-dances-library"

INTENTS: dict[str, tuple[str, ...]] = {
    "happy": ("laughing2", "laughing1"),
    "excited": ("success1", "success2", "laughing1"),
    "loving": ("loving1",),
    "grateful": ("grateful1",),
    "success": ("success1", "success2"),
    "thinking": ("thoughtful1", "thoughtful2"),
    "attentive": ("attentive1", "attentive2"),
    "confused": ("confused1",),
    "uncertain": ("uncertain1",),
    "sad": ("sad1", "sad2", "downcast1"),
    "lonely": ("lonely1",),
    "angry": ("rage1", "irritated2"),
    "irritated": ("irritated1", "irritated2"),
    "disgusted": ("disgusted1",),
    "scared": ("scared1", "fear1"),
    "anxious": ("anxiety1", "fear1"),
    "surprised": ("surprised1", "surprised2", "amazed1"),
    "amazed": ("amazed1", "surprised1"),
    "relief": ("relief1", "relief2"),
    "bored": ("boredom2", "boredom1"),
    "tired": ("exhausted1", "sleep1"),
    "sleepy": ("sleep1", "exhausted1"),
    "yes": ("yes1", "understanding2"),
    "no": ("no1",),
    "welcoming": ("welcoming2",),
    "greeting": ("welcoming2",),
    "goodbye": ("loving1", "welcoming2"),
    "helpful": ("helpful1",),
    "shy": ("shy1",),
}
EMOTION_NAMES = ["random"] + sorted(INTENTS)

DIRECTIONS = {
    "left": (0, 40),
    "right": (0, -40),
    "up": (-30, 0),
    "down": (30, 0),
    "front": (0, 0),
}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


class Motion:
    def __init__(self, mini, stop: threading.Event, enabled=lambda: True, dance_enabled=lambda: False):
        self.mini, self._stop, self._enabled, self._dance_enabled = mini, stop, enabled, dance_enabled
        self._q: queue.Queue = queue.Queue(maxsize=1)
        self._libs: dict[str, object] = {}
        self._lib_err: dict[str, str] = {}
        self._lock = threading.Lock()
        self.busy = False
        self.last = ""
        self.error = ""
        self.recent: collections.deque = collections.deque(maxlen=5)
        threading.Thread(target=self._worker, daemon=True, name="motion").start()
        threading.Thread(target=self._prefetch, daemon=True, name="motion-prefetch").start()

    # -------------------------------------------------------------- libraries ---
    def _prefetch(self) -> None:
        for repo in (EMOTIONS_REPO, DANCES_REPO):
            self._lib(repo)

    def _lib(self, repo: str):
        with self._lock:
            if repo in self._libs:
                return self._libs[repo]
            if repo in self._lib_err and time.time() - float(self._lib_err.get(repo + "_t", 0)) < 60:
                return None
        try:
            from reachy_mini.motion.recorded_move import RecordedMoves

            lib = RecordedMoves(repo)
            with self._lock:
                self._libs[repo] = lib
                self._lib_err.pop(repo, None)
            log.info("Loaded move library %s (%d moves)", repo, len(lib.list_moves()))
            return lib
        except Exception as e:
            log.warning("Move library %s unavailable: %s", repo, e)
            with self._lock:
                self._lib_err[repo] = f"{type(e).__name__}: {e}"[:160]
                self._lib_err[repo + "_t"] = str(time.time())
            return None

    def dance_names(self) -> list[str]:
        lib = self._libs.get(DANCES_REPO)
        try:
            return sorted(lib.list_moves()) if lib else []
        except Exception:
            return []

    def status(self) -> dict:
        return {
            "enabled": bool(self._enabled()),
            "busy": self.busy,
            "last": self.last,
            "emotions_ready": EMOTIONS_REPO in self._libs,
            "dances_ready": DANCES_REPO in self._libs,
            "error": self.error or "; ".join(v for k, v in self._lib_err.items() if not k.endswith("_t")),
        }

    # ------------------------------------------------------------------ worker ---
    def _worker(self) -> None:
        while not self._stop.is_set():
            try:
                label, fn = self._q.get(timeout=0.5)
            except queue.Empty:
                continue
            self.busy, self.last = True, label
            try:
                fn()
                self.error = ""
            except Exception as e:
                self.error = f"{type(e).__name__}: {e}"[:160]
                log.exception("motion %s failed", label)
            finally:
                self.busy = False

    def _submit(self, label: str, fn) -> None:
        if not self._enabled():
            log.debug("Movement is turned off in the settings; ignoring move: %s", label)
            return
        item = (label, fn)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    # ------------------------------------------------------------------- actions ---
    def move_head(self, direction: str) -> str:
        d = DIRECTIONS.get(str(direction).lower())
        if d is None:
            raise ValueError("direction must be left, right, up, down or front")
        pitch, yaw = d

        def run():
            from reachy_mini.utils import create_head_pose

            self.mini.goto_target(head=create_head_pose(pitch=pitch, yaw=yaw, degrees=True), duration=1.0)

        self._submit(f"look {direction}", run)
        return f"looking {direction}"

    def _resolve_emotion(self, requested: str, available: list[str]) -> str | None:
        key = _norm(requested)
        by_key = {_norm(n): n for n in available}
        if key in by_key:
            return by_key[key]
        for cand in INTENTS.get(key, ()):
            if cand in available:
                return cand
        return None

    def emotion(self, requested: str = "random") -> str:
        lib = self._lib(EMOTIONS_REPO)
        if lib is None:
            raise RuntimeError("The built-in emotion moves are not available right now.")
        names = lib.list_moves()
        if not names:
            raise RuntimeError("No emotion moves found.")
        name = self._resolve_emotion(requested or "random", names)
        if not name:
            curated = [m for intents in INTENTS.values() for m in intents if m in names]
            name = random.choice(curated or names)

        def run():
            self.mini.play_move(lib.get(name), initial_goto_duration=0.8)

        self._submit(f"emotion {name}", run)
        return name

    def dance(self, requested: str = "") -> str:
        if not self._dance_enabled():
            log.info("Dancing is disabled in settings; playing expressive emotion instead")
            return self.emotion("happy")

        lib = self._lib(DANCES_REPO)
        if lib is None:
            raise RuntimeError("The built-in dance moves are not available right now.")
        names = lib.list_moves()
        if not names:
            raise RuntimeError("No dance moves found.")
        key = _norm(requested or "")
        by_key = {_norm(n): n for n in names}
        name = by_key.get(key) or next((n for k, n in by_key.items() if key and key in k), None) or random.choice(names)

        def run():
            self.mini.play_move(lib.get(name), initial_goto_duration=1.0)

        self._submit(f"dance {name}", run)
        return name
