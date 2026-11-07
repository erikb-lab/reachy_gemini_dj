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

# Gamepad drive ("manual" mode): the controller page streams a target pose; a 50 Hz loop walks the motors towards it
# at a capped speed, so a dropped packet or a "centre" press can never make the robot jump.
# Units: x/y/z in mm (x forward, y left, z up), angles in degrees (yaw + = left, pitch + = down, roll + = tilt right).
DRIVE_KEYS = ("x", "y", "z", "roll", "pitch", "yaw", "body", "ant_l", "ant_r")
DRIVE_LIMITS = {"x": (-15.0, 15.0), "y": (-15.0, 15.0), "z": (-15.0, 20.0),
                "roll": (-25.0, 25.0), "pitch": (-30.0, 30.0), "yaw": (-60.0, 60.0),
                "body": (-150.0, 150.0),
                # Antennas are asymmetric: across Pollen's recorded emotions and dances the left one stays within
                # -60..+194 deg and the right one within -182..+60 deg. Past those it can hit a stop and the motor
                # shuts itself off with an overload error, so stay well inside.
                "ant_l": (-45.0, 150.0), "ant_r": (-150.0, 45.0)}
DRIVE_SPEED = {"x": 60.0, "y": 60.0, "z": 60.0, "roll": 120.0, "pitch": 120.0, "yaw": 120.0,
               "body": 90.0, "ant_l": 120.0, "ant_r": 120.0}   # max change per second
DRIVE_HZ = 50.0
DRIVE_IDLE_S = 180.0      # no input for this long: leave manual mode (the robot keeps holding the pose)


def head_pose(x_mm: float, y_mm: float, z_mm: float, roll: float, pitch: float, yaw: float):
    """4x4 head pose, same convention as reachy_mini.utils.create_head_pose (extrinsic xyz Euler, degrees, mm)."""
    import math
    import numpy as np
    r, p, y = math.radians(roll), math.radians(pitch), math.radians(yaw)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    m = np.eye(4)
    m[:3, :3] = [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                 [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                 [-sp, cp * sr, cp * cr]]
    m[:3, 3] = [x_mm / 1000.0, y_mm / 1000.0, z_mm / 1000.0]
    return m


def matrix_to_rpy(m) -> tuple[float, float, float]:
    """Inverse of head_pose's rotation: (roll, pitch, yaw) in degrees."""
    import math
    pitch = math.asin(max(-1.0, min(1.0, -float(m[2][0]))))
    roll = math.atan2(float(m[2][1]), float(m[2][2]))
    yaw = math.atan2(float(m[1][0]), float(m[0][0]))
    return math.degrees(roll), math.degrees(pitch), math.degrees(yaw)


def drive_clamp(target: dict) -> dict:
    """Keep only known keys with finite numbers, clamped to the safe range."""
    out = {}
    for k, v in (target or {}).items():
        if k not in DRIVE_LIMITS:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or f in (float("inf"), float("-inf")):
            continue
        lo, hi = DRIVE_LIMITS[k]
        out[k] = min(hi, max(lo, f))
    return out


def drive_step(current: dict, wanted: dict, dt: float) -> dict:
    """Move each value of current towards wanted by at most DRIVE_SPEED * dt."""
    out = dict(current)
    for k in DRIVE_KEYS:
        if k not in wanted:
            continue
        step = DRIVE_SPEED[k] * dt
        d = wanted[k] - out.get(k, 0.0)
        out[k] = out.get(k, 0.0) + max(-step, min(step, d))
    return out


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
        self.manual = False                    # gamepad drive mode: other moves are ignored while it is on
        self._drive_cmd: dict = {}             # what the motors were last told
        self._drive_want: dict = {}            # what the page asked for
        self._drive_seen = 0.0                 # last input from the page (monotonic)
        from .poses import PoseStore
        self.poses = PoseStore()
        threading.Thread(target=self._worker, daemon=True, name="motion").start()
        threading.Thread(target=self._prefetch, daemon=True, name="motion-prefetch").start()
        threading.Thread(target=self._drive_loop, daemon=True, name="motion-drive").start()

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

    def moving(self) -> bool:
        """True while Reachy moves itself or is limp for hand posing (the IMU watcher must ignore the sensor then)."""
        return bool(self.busy or self.learning or self.manual or not self._q.empty())

    def status(self) -> dict:
        return {
            "enabled": bool(self._enabled()),
            "busy": self.busy,
            "last": self.last,
            "emotions_ready": EMOTIONS_REPO in self._libs,
            "dances_ready": DANCES_REPO in self._libs,
            "error": self.error or "; ".join(v for k, v in self._lib_err.items() if not k.endswith("_t")),
            "learning": self.learning,
            "manual": self.manual,
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
        if self.manual:
            log.info("Gamepad mode is on; ignoring move: %s", label)
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
            if self.learning or self.manual or not self._enabled() or self.busy or not self._q.empty():
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
        if self.manual:
            self.drive_stop()
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
        if self.manual:
            self.drive_stop()

        def run():
            import numpy as np
            joints = p.get("head_joints") or []
            self.mini.goto_target(head=np.array(p["head"], dtype=float), antennas=p.get("antennas"),
                                  duration=duration, body_yaw=float(joints[0]) if joints else 0.0)

        self._submit(f"pose {real}", run)
        return real

    # ------------------------------------------------------------- gamepad drive ---
    def _read_drive_pose(self, clamp: bool = True) -> dict:
        """The robot's present pose in drive units (falls back to neutral if it cannot be read)."""
        pose = {k: 0.0 for k in DRIVE_KEYS}
        try:
            import math
            m = self.mini.get_current_head_pose()
            roll, pitch, yaw = matrix_to_rpy(m)
            head_joints, antennas = self.mini.get_current_joint_positions()
            pose.update(x=float(m[0][3]) * 1000, y=float(m[1][3]) * 1000, z=float(m[2][3]) * 1000,
                        roll=float(roll), pitch=float(pitch), yaw=float(yaw))
            if head_joints:
                pose["body"] = math.degrees(float(head_joints[0]))
            if antennas and len(antennas) == 2:
                pose["ant_r"], pose["ant_l"] = math.degrees(float(antennas[0])), math.degrees(float(antennas[1]))
        except Exception as e:
            log.warning("could not read the pose for gamepad mode: %s", e)
        return drive_clamp(pose) if clamp else pose

    def drive_start(self) -> dict:
        """Gamepad mode on: motors on, other moves paused, start from where the robot is now."""
        if self.manual:
            return self.drive_state()
        if self.learning:
            self.learn_stop()
        try:                                   # drop a queued move and let a running one finish
            self._q.get_nowait()
        except queue.Empty:
            pass
        deadline = time.monotonic() + 8.0
        while self.busy and time.monotonic() < deadline:
            time.sleep(0.05)
        self._ensure_motors()
        pose = self._read_drive_pose(clamp=False)
        with self._lock:
            self._drive_cmd = dict(pose)               # where it really is (may be outside the safe range)...
            self._drive_want = drive_clamp(pose)       # ...walked back inside at the capped speed, never jumped
            self._drive_seen = time.monotonic()
            self._thinking = self._tilted = False
            self.manual = True
        self.last = "gamepad mode"
        log.info("gamepad mode on")
        return self.drive_state()

    def drive_stop(self) -> dict:
        """Gamepad mode off. The robot keeps holding the last pose; normal moves resume."""
        if self.manual:
            log.info("gamepad mode off")
        self.manual = False
        self.last = "pose held"
        return self.drive_state()

    def drive_set(self, target: dict) -> dict:
        if not self.manual:
            raise RuntimeError("Gamepad mode is off.")
        clean = drive_clamp(target)
        with self._lock:
            self._drive_want.update(clean)
            self._drive_seen = time.monotonic()
        return self.drive_state()

    def drive_state(self) -> dict:
        with self._lock:
            return {"manual": self.manual, "target": dict(self._drive_want),
                    "limits": {k: list(v) for k, v in DRIVE_LIMITS.items()}}

    def _drive_send(self, cmd: dict) -> None:
        import math
        head = head_pose(cmd["x"], cmd["y"], cmd["z"], cmd["roll"], cmd["pitch"], cmd["yaw"])
        self.mini.set_target(head=head, antennas=[math.radians(cmd["ant_r"]), math.radians(cmd["ant_l"])],
                             body_yaw=math.radians(cmd["body"]))

    def _drive_loop(self) -> None:
        dt = 1.0 / DRIVE_HZ
        last_err = 0.0
        while not self._stop.is_set():
            if not self.manual:
                time.sleep(0.1)
                continue
            if time.monotonic() - self._drive_seen > DRIVE_IDLE_S:
                log.info("gamepad mode: no input for %.0f s, leaving it", DRIVE_IDLE_S)
                self.drive_stop()
                continue
            with self._lock:
                nxt = drive_step(self._drive_cmd, self._drive_want, dt)
                changed = nxt != self._drive_cmd
                self._drive_cmd = nxt
            if changed:
                try:
                    self._drive_send(nxt)
                except Exception as e:
                    if time.monotonic() - last_err > 5:
                        log.warning("gamepad drive: %s", e)
                        self.error = f"{type(e).__name__}: {e}"[:160]
                        last_err = time.monotonic()
            time.sleep(dt)
