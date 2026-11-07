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


THINK_TILT_DEG = 9.0      # sideways head roll for the "thinking" tilt
THINK_TILT_S = 1.4        # slow on purpose


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
        self.learning = False
        self._thinking = False                 # wanted: head tilted while the brain thinks
        self._tilted = False                   # the head is actually tilted
        from .poses import PoseStore
        self.poses = PoseStore()
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
            "learning": self.learning,
            "poses": self.poses.names(),
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
        if self.learning:
            log.info("Learn mode is on (motors relaxed); ignoring move: %s", label)
            return
        if not self._enabled():
            log.debug("Movement is turned off in the settings; ignoring move: %s", label)
            return
        def powered(fn=fn):
            self._ensure_motors()
            fn()
        item = (label, powered)
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def _ensure_motors(self) -> None:
        """The robot daemon can start (or be left) with motors off; a move would then silently do nothing."""
        try:
            self.mini.enable_motors()
        except Exception as e:
            log.warning("could not enable motors: %s", e)

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

    def thinking(self, on: bool) -> None:
        """Subtle, slow, curious head tilt while the local brain works on a turn; back to neutral when it is over.

        Cheap (flag checks only) and polite: skipped if motion is off, in learn mode, or a move/dance is running or
        queued (the tilt is never allowed to interrupt or replace one). Safe to call repeatedly with the same value.
        """
        if on:
            if self._thinking:
                return
            if self.learning or not self._enabled() or self.busy or not self._q.empty():
                return
            self._thinking = True

            def tilt():
                if not self._thinking:             # the turn ended while this was waiting in the queue
                    return
                from reachy_mini.utils import create_head_pose
                self.mini.goto_target(head=create_head_pose(roll=THINK_TILT_DEG, pitch=-3, yaw=4, degrees=True),
                                      duration=THINK_TILT_S)
                self._tilted = True
                if not self._thinking:             # the turn ended while the head was tilting: level it right away
                    self.mini.goto_target(head=create_head_pose(), duration=1.0)
                    self._tilted = False

            self._submit("thinking", tilt)
            return
        self._thinking = False
        if not self._tilted:
            return
        if self.learning or self.busy or not self._q.empty():
            self._tilted = False                   # another move is running or queued and will move the head anyway
            return

        def level():
            if self._thinking or not self._tilted:
                return
            from reachy_mini.utils import create_head_pose
            self.mini.goto_target(head=create_head_pose(), duration=1.0)
            self._tilted = False

        self._submit("thinking done", level)

    # ------------------------------------------------------------- learn poses ---
    def learn_start(self) -> str:
        """Turn the motors off so the robot can be posed freely by hand (the head will not hold itself up)."""
        if self.learning:
            return "already in learn mode"
        while self.busy:                       # let a running move finish before going limp
            time.sleep(0.05)
        self.mini.disable_motors()
        self.learning = True
        self.last = "learn mode (motors off)"
        return "motors off: pose me by hand, then lock"

    def learn_stop(self) -> str:
        """Turn the motors back on and lock the robot where it was left (no snap back to the old target)."""
        if not self.learning:
            return "learn mode is off"
        head = antennas = body_yaw = None
        try:
            head = self.mini.get_current_head_pose()
            head_joints, antennas = self.mini.get_current_joint_positions()
            body_yaw = float(head_joints[0]) if head_joints else None
        except Exception as e:
            log.warning("could not read the pose before locking: %s", e)
        if head is not None:                   # aim the controller at the current position first...
            try:
                self.mini.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            except Exception as e:
                log.debug("pre-lock target: %s", e)
        self.mini.enable_motors()              # ...then power the motors so they hold it
        if head is not None:
            try:
                self.mini.set_target(head=head, antennas=antennas, body_yaw=body_yaw)
            except Exception as e:
                log.debug("lock target: %s", e)
        self.learning = False
        self.last = "pose locked"
        return "motors on: pose locked"

    def capture(self) -> dict:
        head = self.mini.get_current_head_pose()
        head_joints, antennas = self.mini.get_current_joint_positions()
        return {"head": [[float(x) for x in row] for row in head],
                "antennas": [float(a) for a in antennas],
                "head_joints": [float(j) for j in head_joints]}

    def save_pose(self, name: str) -> str:
        """Lock the pose if the motors are off, then read it and write it to poses.json immediately."""
        from .poses import clean_name
        clean_name(name)                       # reject an empty name before touching the motors
        if self.learning:
            self.learn_stop()
            time.sleep(0.3)                    # let the motors settle on the locked pose
        return self.poses.save(name, self.capture())

    def goto_pose(self, name: str, duration: float = 1.2) -> str:
        found = self.poses.find(name)
        if not found:
            names = self.poses.names()
            raise ValueError(f"No saved pose called '{name}'." + (f" Saved poses: {', '.join(names)}." if names else
                                                                    " No poses saved yet."))
        real, p = found
        if self.learning:
            self.learn_stop()

        def run():
            import numpy as np
            joints = p.get("head_joints") or []
            self.mini.goto_target(head=np.array(p["head"], dtype=float), antennas=p.get("antennas"),
                                  duration=duration, body_yaw=float(joints[0]) if joints else 0.0)

        self._submit(f"pose {real}", run)
        return real
