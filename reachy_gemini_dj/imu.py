"""Reachy's motion sensor (IMU): pure helpers, an event watcher and spoken reactions.

`ReachyMini.imu` gives {"accelerometer": [x, y, z] m/s^2, "gyroscope": [x, y, z] rad/s, "quaternion": [w, x, y, z],
"temperature": deg C} (or None on a Lite / simulator / stale). Axis z points up on the robot at rest (a level robot reads
about +9.8 on z), so tilt is the angle between z and the measured gravity vector.

ImuWatcher polls the sensor (~20 Hz), keeps a smoothed state and detects picked up / set down / bumped / shaken / tipped
over. ImuReactions turns those events into a short spoken line (and sometimes a small move). Everything takes an injected
clock and sensor reader, so the tests feed synthetic streams. Nothing here touches audio content; detections are logged
at INFO with the numbers so the thresholds can be tuned on the real robot.
"""
from __future__ import annotations

import collections
import logging
import math
import random
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------------------------- thresholds ---
G = 9.80665                  # standard gravity, m/s^2
POLL_HZ = 20.0
STALE_S = 3.0                # no good sample for this long: the sensor reading is "not available"
NO_DATA_SLOW_AFTER = 20      # this many empty reads in a row (Lite / simulator): poll once a second instead
# smoothing time constants (seconds)
TAU_VEC = 0.30               # gravity vector (tilt)
TAU_GYRO = 0.30              # |gyro|
TAU_DYN = 0.25               # | |a| - baseline |
TAU_BASE = 4.0               # slow baseline of |a| (a real unit reads ~9.68, not 9.81, so never compare to 1 g directly)
TAU_HP = 1.0                 # slow accel vector, subtracted to get the shake's alternating part
# "still" and "active" (smoothed values). Rest noise measured on the robot: |gyro| ~0.002 rad/s, |a| sd ~0.02 m/s^2.
STILL_GYRO = 0.06            # rad/s
STILL_DYN = 0.15             # m/s^2
ACT_GYRO = 0.30              # rad/s: being carried / handled
ACT_DYN = 0.80               # m/s^2
# picked up / set down
PICKUP_MIN_S = 0.8           # continuous activity before it counts as a pickup
PICKUP_RAW_FRAC = 0.7        # ... and this share of the raw samples in that time must be active (a knock rings on in the smoothing)
PICKUP_TILT_DEG = 6.0        # ... plus this much change from the resting orientation (or a real acceleration, below)
PICKUP_DYN = 0.8             # m/s^2 peak smoothed deviation while active
SETDOWN_STILL_S = 1.5        # still this long after being held = set down
HELD_MIN_S = 2.0             # a "ride" shorter than this gets no set-down line
# bump / tap: a short spike while otherwise calm
TAP_ACC = 0.8                # m/s^2 raw |a| deviation from baseline
TAP_GYRO = 0.6               # rad/s raw
TAP_CALM_S = 0.7             # must have been calm this long before the spike
TAP_CALM_GYRO = 0.12         # smoothed values just before the spike
TAP_CALM_DYN = 0.25
TAP_WINDOW_S = 0.6           # judged this long after the spike starts
TAP_MAX_ABOVE_S = 0.30       # a spike lasting longer than this is a shove, not a tap
TAP_END_GYRO = 0.25          # still this calm at the end of the window
# shake
SHAKE_GYRO = 2.0             # rad/s raw
SHAKE_ACC = 3.0              # m/s^2 raw deviation
SHAKE_WINDOW_S = 1.5
SHAKE_SPAN_S = 1.0           # first to last violent sample at least this far apart ("for >= 1 s")
SHAKE_MIN_HIGH = 10          # violent samples in the window
SHAKE_MIN_REVERSALS = 4      # direction changes of the alternating acceleration
SHAKE_REV_MIN = 1.5          # m/s^2: both halves of a reversal must be at least this large
SHAKE_PRE = 6                # this many violent samples in the window postpones a pickup (it is probably a shake)
# tipped over
TIP_DEG = 60.0
TIP_CLEAR_DEG = 50.0
TIP_MIN_S = 2.0
TIP_MAX_GYRO = 0.15
TIP_REPEAT_S = 30.0
TIP_MAX_ANNOUNCE = 2
# posture bands with hysteresis: (enter above, leave below)
LEAN_ENTER, LEAN_LEAVE = 28.0, 22.0
SIDE_ENTER, SIDE_LEAVE = 65.0, 55.0
UPSIDE_ENTER, UPSIDE_LEAVE = 125.0, 115.0
MOVING_GYRO = 0.15           # snapshot "moving" (looser than ACT_*: any real motion)
MOVING_DYN = 0.40
# per-event cooldowns (seconds) and general guards
COOLDOWN_S = {"picked_up": 20.0, "set_down": 20.0, "bumped": 20.0, "shaken": 20.0, "tipped": 20.0}
OWN_MOVE_GRACE_S = 1.5       # ignore the sensor this long after Reachy's own move
WARMUP_S = 3.0               # no events right after start (baselines settling)
GAP_RESET_S = 1.0            # a gap between samples this long restarts the detectors
HOT_C = 70.0

EVENTS = ("picked_up", "set_down", "bumped", "shaken", "tipped")


# ----------------------------------------------------------------------------------------------- helpers ---
def _vec(v: Any) -> tuple[float, float, float] | None:
    try:
        x, y, z = (float(c) for c in v)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(c) for c in (x, y, z)):
        return None
    return x, y, z


def norm(v: tuple[float, float, float]) -> float:
    return math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])


def tilt_deg(accel: Any) -> float | None:
    """Angle in degrees between the robot's up axis (z) and straight up, from the gravity reading. None if unusable."""
    a = _vec(accel)
    if a is None:
        return None
    n = norm(a)
    if n < 1.0:                            # free fall / garbage: gravity direction unknown
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0, a[2] / n))))


def angle_between(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    na, nb = norm(a), norm(b)
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    c = (a[0] * b[0] + a[1] * b[1] + a[2] * b[2]) / (na * nb)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def posture_for_tilt(tilt: float, prev: str = "") -> str:
    """'upright' | 'leaning' | 'on its side' | 'upside down'. Pass the previous word to get hysteresis (no flicker)."""
    order = ["upright", "leaning", "on its side", "upside down"]
    enter = [0.0, LEAN_ENTER, SIDE_ENTER, UPSIDE_ENTER]
    leave = [0.0, LEAN_LEAVE, SIDE_LEAVE, UPSIDE_LEAVE]
    i = order.index(prev) if prev in order else None
    if i is None:
        i = max(k for k in range(4) if tilt >= enter[k])
        return order[i]
    while i < 3 and tilt >= enter[i + 1]:
        i += 1
    while i > 0 and tilt < leave[i]:
        i -= 1
    return order[i]


def motion_magnitude(gyro: Any) -> float:
    """|angular velocity| in rad/s (0.0 if unusable)."""
    g = _vec(gyro)
    return norm(g) if g else 0.0


def accel_deviation_g(accel: Any) -> float:
    """| |a| - 1 g | in units of g (0.0 if unusable). A still unit reads within ~0.02 of 1 g."""
    a = _vec(accel)
    return abs(norm(a) - G) / G if a else 0.0


def _alpha(dt: float, tau: float) -> float:
    return 1.0 - math.exp(-dt / tau)


def _round_temp(t: Any) -> float | None:
    try:
        f = float(t)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ------------------------------------------------------------------------------------------- the watcher ---
class ImuWatcher:
    """Polls `read_imu()` (a dict as above, or None) and detects events. `on_event(kind, info)` is called from the
    watcher thread. `suppress()` -> True while Reachy moves itself, dances, is in learn mode or quiet mode: everything
    is ignored then and for OWN_MOVE_GRACE_S afterwards. `no_bump()` -> True while the robot itself is loud (music,
    its own voice): taps are not detected then (speaker vibration looks like taps)."""

    def __init__(self, read_imu: Callable[[], Any], clock: Callable[[], float] = time.monotonic,
                 on_event: Callable[[str, dict], None] | None = None,
                 suppress: Callable[[], bool] | None = None, no_bump: Callable[[], bool] | None = None,
                 hz: float = POLL_HZ):
        self.read_imu = read_imu
        self.clock = clock
        self.on_event = on_event or (lambda kind, info: None)
        self.suppress = suppress or (lambda: False)
        self.no_bump = no_bump or (lambda: False)
        self.period = 1.0 / hz
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nones = 0
        self._warned = False
        self.events: collections.deque = collections.deque(maxlen=20)
        self._start_t: float | None = None
        self._cool: dict[str, float] = {}
        # smoothed state (first sample seeds it)
        self._t_last: float | None = None
        self._t_good = -1e9
        self._seeded = False
        self._vec_s = (0.0, 0.0, G)
        self._vec_slow = (0.0, 0.0, G)
        self._base = G
        self._gyro_s = 0.0
        self._dyn_s = 0.0
        self._tilt = 0.0
        self._posture = "upright"
        self._raw_norm = G
        self._raw_gyro = 0.0
        self._temp: float | None = None
        self._grace_until = -1e9
        self._reset_detectors()

    # ----------------------------------------------------------------------------------------- detectors ---
    def _reset_detectors(self) -> None:
        self._still_since: float | None = None
        self._rest_vec: tuple[float, float, float] | None = None
        self._active_since: float | None = None
        self._last_active = -1e9
        self._peak_dyn = 0.0
        self._held = False
        self._held_since = 0.0
        self._ride_announced = False
        self._calm_since: float | None = None
        self._rawact: collections.deque = collections.deque()      # (t, raw sample looked active)
        self._window: collections.deque = collections.deque()      # (t, violent, hp_vec)
        self._prev_hp: tuple[float, float, float] | None = None
        self._reversals: collections.deque = collections.deque()   # timestamps
        self._tap: dict | None = None
        self._tip_since: float | None = None
        self._tip_n = 0
        self._tip_last = -1e9

    # ---------------------------------------------------------------------------------------- the sample ---
    def feed(self, imu: Any) -> None:
        """Process one sensor reading taken now (clock()). Public so tests can drive it with synthetic data."""
        with self._lock:
            self._feed(imu)

    def _feed(self, imu: Any) -> None:
        t = self.clock()
        try:
            a = _vec(imu.get("accelerometer")) if isinstance(imu, dict) else None
            w = _vec(imu.get("gyroscope")) if isinstance(imu, dict) else None
        except Exception:
            a = w = None
        if a is None or w is None or norm(a) < 1.0:
            return                                              # no / unusable sample: state goes stale by itself
        if self._start_t is None:
            self._start_t = t
        temp = _round_temp(imu.get("temperature"))
        if temp is not None:
            self._temp = temp
        gap = None if self._t_last is None else t - self._t_last
        self._t_last = t
        self._t_good = t
        if not self._seeded or (gap is not None and gap > GAP_RESET_S):
            self._seed(a, w)
            return
        dt = max(1e-3, min(gap or self.period, GAP_RESET_S))

        n, gm = norm(a), norm(w)
        prev_gyro, prev_dyn = self._gyro_s, self._dyn_s
        raw_dev = abs(n - self._base)
        # smoothing
        av = _alpha(dt, TAU_VEC)
        self._vec_s = tuple(s + av * (c - s) for s, c in zip(self._vec_s, a))
        hp = tuple(c - s for c, s in zip(a, self._vec_slow))
        ah = _alpha(dt, TAU_HP)
        self._vec_slow = tuple(s + ah * (c - s) for s, c in zip(self._vec_slow, a))
        self._gyro_s += _alpha(dt, TAU_GYRO) * (gm - self._gyro_s)
        self._dyn_s += _alpha(dt, TAU_DYN) * (raw_dev - self._dyn_s)
        if raw_dev < 0.5:                                       # do not let the baseline chase real motion
            self._base += _alpha(dt, TAU_BASE) * (n - self._base)
        self._raw_norm, self._raw_gyro = n, gm
        tl = tilt_deg(self._vec_s)
        if tl is not None:
            self._tilt = tl
            self._posture = posture_for_tilt(tl, self._posture)

        # own movement / quiet mode / learn mode: ignore everything, restart cleanly afterwards
        try:
            if self.suppress():
                self._grace_until = t + OWN_MOVE_GRACE_S
        except Exception:
            log.debug("imu suppress() failed", exc_info=True)
        if t < self._grace_until:
            self._reset_detectors()
            self._base = n
            return
        if t - (self._start_t or t) < WARMUP_S:
            return

        still = self._gyro_s < STILL_GYRO and self._dyn_s < STILL_DYN
        active = self._gyro_s > ACT_GYRO or self._dyn_s > ACT_DYN
        if still:
            if self._still_since is None:
                self._still_since = t
            if t - self._still_since >= 1.0:
                self._rest_vec = self._vec_s
        else:
            self._still_since = None
        last_active_before = self._last_active
        if active:
            if self._active_since is None:
                self._active_since = t
                self._peak_dyn = 0.0
            self._peak_dyn = max(self._peak_dyn, self._dyn_s)
            self._last_active = t
        else:
            self._active_since = None

        self._rawact.append((t, gm > ACT_GYRO or raw_dev > ACT_DYN))
        while self._rawact and t - self._rawact[0][0] > PICKUP_MIN_S:
            self._rawact.popleft()
        violent = gm > SHAKE_GYRO or raw_dev > SHAKE_ACC
        self._shake_window(t, violent, hp)
        if self._check_shake(t):
            return
        self._check_pickup_setdown(t, still)
        self._check_tap(t, raw_dev, gm, prev_gyro, prev_dyn, last_active_before)
        self._check_tipped(t)

    def _seed(self, a, w) -> None:
        self._vec_s = self._vec_slow = a
        self._base = norm(a)
        self._gyro_s = norm(w)
        self._dyn_s = 0.0
        self._raw_norm, self._raw_gyro = norm(a), norm(w)
        tl = tilt_deg(a)
        self._tilt = tl if tl is not None else 0.0
        self._posture = posture_for_tilt(self._tilt)
        self._seeded = True
        self._reset_detectors()

    # ---------------------------------------------------------------------------------------------- shake ---
    def _shake_window(self, t: float, violent: bool, hp) -> None:
        w = self._window
        w.append((t, violent, hp))
        while w and t - w[0][0] > SHAKE_WINDOW_S:
            w.popleft()
        if violent and self._prev_hp is not None:
            dot = sum(p * c for p, c in zip(self._prev_hp, hp))
            if dot < 0 and norm(hp) >= SHAKE_REV_MIN and norm(self._prev_hp) >= SHAKE_REV_MIN:
                self._reversals.append(t)
        if violent:
            self._prev_hp = hp
        while self._reversals and t - self._reversals[0] > SHAKE_WINDOW_S:
            self._reversals.popleft()

    def _violent_times(self) -> list[float]:
        return [x[0] for x in self._window if x[1]]

    def _check_shake(self, t: float) -> bool:
        v = self._violent_times()
        if len(v) < SHAKE_MIN_HIGH or v[-1] - v[0] < SHAKE_SPAN_S or len(self._reversals) < SHAKE_MIN_REVERSALS:
            return False
        info = {"violent_samples": len(v), "reversals": len(self._reversals), "span_s": round(v[-1] - v[0], 2),
                "gyro": round(self._raw_gyro, 2), "dyn": round(self._dyn_s, 2)}
        self._window.clear()
        self._reversals.clear()
        self._prev_hp = None
        self._tap = None
        if not self._held:                                      # being shaken means being held; no pickup/ride lines
            self._held, self._held_since, self._ride_announced = True, t, False
        self._emit("shaken", t, info)
        return True

    # ------------------------------------------------------------------------------ pickup / set down ---
    def _violent_recent(self) -> int:
        return sum(1 for x in self._window if x[1])

    def _check_pickup_setdown(self, t: float, still: bool) -> None:
        if not self._held:
            if self._active_since is None or t - self._active_since < PICKUP_MIN_S:
                return
            ra = self._rawact
            if len(ra) < 2 or sum(1 for x in ra if x[1]) / len(ra) < PICKUP_RAW_FRAC:
                return                                          # a knock that the smoothing rings on after
            if self._violent_recent() >= SHAKE_PRE:
                return                                          # probably a shake: wait for it to be judged
            change = angle_between(self._vec_s, self._rest_vec) if self._rest_vec else 0.0
            if change < PICKUP_TILT_DEG and self._peak_dyn < PICKUP_DYN:
                return
            self._held, self._held_since = True, t
            self._tap = None
            self._ride_announced = self._emit("picked_up", t, {"tilt_change": round(change, 1), "peak_dyn": round(self._peak_dyn, 2),
                                                               "gyro": round(self._gyro_s, 2), "tilt": round(self._tilt, 1)})
            return
        if not still:
            self._calm_since = None
            return
        if self._calm_since is None:
            self._calm_since = t
        if t - self._calm_since >= SETDOWN_STILL_S:
            held_for = t - self._held_since
            announced = self._ride_announced
            self._held = False
            self._ride_announced = False
            self._calm_since = None
            self._rest_vec = self._vec_s
            if announced and held_for >= HELD_MIN_S:
                self._emit("set_down", t, {"held_s": round(held_for, 1), "tilt": round(self._tilt, 1)})

    # -------------------------------------------------------------------------------------------- bump ---
    def _check_tap(self, t: float, raw_dev: float, gm: float, prev_gyro: float, prev_dyn: float,
                   last_active: float) -> None:
        if self._held:
            self._tap = None
            return
        try:
            loud = bool(self.no_bump())
        except Exception:
            loud = False
        if self._tap is None:
            spike = raw_dev > TAP_ACC or gm > TAP_GYRO
            calm = (prev_gyro < TAP_CALM_GYRO and prev_dyn < TAP_CALM_DYN and t - last_active >= TAP_CALM_S)
            if spike and calm and not loud:
                self._tap = {"t0": t, "peak": raw_dev, "gpeak": gm, "above": 0.0, "last": t}
            return
        tp = self._tap
        dt = t - tp["last"]
        tp["last"] = t
        tp["peak"], tp["gpeak"] = max(tp["peak"], raw_dev), max(tp["gpeak"], gm)
        if raw_dev > TAP_ACC or gm > TAP_GYRO:
            tp["above"] += dt
        if t - tp["t0"] < TAP_WINDOW_S:
            return
        self._tap = None
        if loud or self._active_since is not None or tp["above"] > TAP_MAX_ABOVE_S or self._gyro_s > TAP_END_GYRO:
            return
        self._emit("bumped", t, {"peak_acc": round(tp["peak"], 2), "peak_gyro": round(tp["gpeak"], 2)})

    # ------------------------------------------------------------------------------------------ tipped ---
    def _check_tipped(self, t: float) -> None:
        if self._tilt < TIP_CLEAR_DEG:
            self._tip_since, self._tip_n = None, 0
            return
        if self._tilt <= TIP_DEG or self._held or self._gyro_s > TIP_MAX_GYRO:
            if self._held or self._gyro_s > TIP_MAX_GYRO:
                self._tip_since = None
            return
        if self._tip_since is None:
            self._tip_since = t
            return
        if t - self._tip_since < TIP_MIN_S or self._tip_n >= TIP_MAX_ANNOUNCE or t - self._tip_last < TIP_REPEAT_S:
            return
        if self._emit("tipped", t, {"tilt": round(self._tilt, 1), "posture": self._posture,
                                    "still_s": round(t - self._tip_since, 1), "n": self._tip_n + 1}):
            self._tip_n += 1
            self._tip_last = t

    # ----------------------------------------------------------------------------------------- emission ---
    def _emit(self, kind: str, t: float, info: dict) -> bool:
        if t - (self._start_t or t) < WARMUP_S:
            return False
        last = self._cool.get(kind)
        if last is not None and t - last < COOLDOWN_S.get(kind, 20.0):
            log.info("imu %s ignored (cooldown): %s", kind, info)
            return False
        self._cool[kind] = t
        self.events.append({"kind": kind, "t": t, **info})
        log.info("imu event %s: %s", kind, info)
        try:
            self.on_event(kind, dict(info))
        except Exception:
            log.exception("imu event handler failed")
        return True

    # --------------------------------------------------------------------------------------------- state ---
    def snapshot(self) -> dict | None:
        """Latest smoothed state, or None when there is no fresh sample (no IMU, or it stopped answering)."""
        with self._lock:
            if not self._seeded or self.clock() - self._t_good > STALE_S:
                return None
            moving = self._gyro_s > MOVING_GYRO or self._dyn_s > MOVING_DYN
            return {"tilt_deg": round(self._tilt, 1), "posture": self._posture, "moving": bool(moving),
                    "held": bool(self._held), "temperature_c": None if self._temp is None else round(self._temp, 1),
                    "gyro_rad_s": round(self._gyro_s, 3), "accel_dev_g": round(abs(self._raw_norm - G) / G, 3)}

    def status(self) -> dict:
        """Cheap dict for /api/state."""
        s = self.snapshot()
        if s is None:
            return {"available": False}
        return {"available": True, **s, "hot": bool(s["temperature_c"] is not None and s["temperature_c"] >= HOT_C)}

    def sensors(self) -> dict:
        """Result of the robot_sensors tool."""
        return describe(self.snapshot())

    # ---------------------------------------------------------------------------------------------- thread ---
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="imu-watcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(self.period if self._nones < NO_DATA_SLOW_AFTER else 1.0):
            try:
                imu = self.read_imu()
            except Exception as e:
                imu = None
                if not self._warned:
                    self._warned = True
                    log.warning("could not read the IMU: %s", e)
            if imu is None:
                self._nones += 1
                continue
            self._nones = 0
            try:
                self.feed(imu)
            except Exception:
                log.exception("imu processing failed")


# --------------------------------------------------------------------------------------------- the tool ---
def describe(snap: dict | None) -> dict:
    """robot_sensors result: compact numbers plus a speakable `say`."""
    if not snap:
        return {"error": "I can't read my motion sensor right now."}
    tilt = float(snap["tilt_deg"])
    posture = snap["posture"]
    if snap.get("held"):
        motion = "it feels like I'm being held"
    elif snap.get("moving"):
        motion = "I'm moving"
    else:
        motion = "I'm not moving"
    if posture == "upright":
        t = int(round(tilt))
        pose = "I'm sitting upright and level" if t < 2 else f"I'm sitting upright, tilted about {t} degrees"
    elif posture == "leaning":
        pose = f"I'm leaning about {int(round(tilt / 5.0) * 5)} degrees"
    elif posture == "on its side":
        pose = "I'm on my side"
    else:
        pose = "I'm upside down"
    say = f"{pose}, and {motion}."
    temp = snap.get("temperature_c")
    out: dict = {"tilt_deg": round(tilt), "posture": posture, "moving": bool(snap.get("moving")),
                 "held": bool(snap.get("held"))}
    if temp is not None:
        deg = int(round(temp))
        say += f" My internal temperature is {deg} degrees Celsius"
        say += ", which is hot." if temp >= HOT_C else "."
        out["temperature_c"] = round(temp, 1)
        out["hot"] = bool(temp >= HOT_C)
    out["say"] = say
    return out


# ------------------------------------------------------------------------------------------ reactions ---
class ImuReactions:
    """Spoken reaction to an IMU event. Never while the robot or the user is speaking (`is_idle`), never in quiet mode,
    never two at once (MIN_GAP_S apart), and the move is only played when motion is free (`motion_free`)."""

    MIN_GAP_S = 4.0

    LINES = {
        "picked_up": ("Whoa, where are we going?",),
        "set_down": ("Thanks for the ride.",),
        "bumped": ("Hey!",),
        "shaken": ("I'm getting dizzy!",),
        "tipped": ("I fell over. Can you help me up?",),
    }
    EMOTIONS = {"picked_up": ("surprised", "attentive")}

    def __init__(self, say: Callable[[str], None], emotion: Callable[[str], Any] | None = None,
                 enabled: Callable[[], bool] = lambda: True, is_idle: Callable[[], bool] = lambda: True,
                 is_quiet: Callable[[], bool] = lambda: False, motion_free: Callable[[], bool] = lambda: True,
                 clock: Callable[[], float] = time.monotonic, rng: random.Random | None = None):
        self.say, self.emotion = say, emotion
        self.enabled, self.is_idle, self.is_quiet, self.motion_free = enabled, is_idle, is_quiet, motion_free
        self.clock = clock
        self.rng = rng or random.Random()
        self._last = -1e9
        self._lock = threading.Lock()

    def on_event(self, kind: str, info: dict | None = None) -> bool:
        lines = self.LINES.get(kind)
        if not lines:
            return False
        try:
            if not self.enabled() or self.is_quiet() or not self.is_idle():
                return False
        except Exception:
            return False
        with self._lock:
            now = self.clock()
            if now - self._last < self.MIN_GAP_S:
                return False
            self._last = now
        try:
            self.say(self.rng.choice(lines))
        except Exception:
            log.exception("imu reaction failed")
            return False
        emotions = self.EMOTIONS.get(kind)
        if emotions and self.emotion is not None:
            try:
                if self.motion_free():
                    self.emotion(self.rng.choice(emotions))
            except Exception:
                log.debug("imu reaction move failed", exc_info=True)
        return True
