"""Small quality-of-life features: the conversation log, the brain status chip, quiet mode and the "I'm ready" greeting.

Everything here is plain Python (no robot, no network) so it can be tested on its own; main.py only wires it up.
The conversation log lives in memory only, on purpose: what people say to the robot is never written to the app log.
"""
from __future__ import annotations

import collections
import logging
import threading
import time
from typing import Any, Callable

log = logging.getLogger(__name__)

MAX_TURNS = 20
MAX_TEXT = 1000
GAP_S = 8.0               # Gemini: a new heard fragment this long after the last one (and no reply yet) is a new turn
LATE_S = 2.0              # transcripts can trail the end-of-turn signal; text this soon after joins the last turn

QUIET_DEFAULT_MIN = 30
QUIET_MAX_MIN = 480


def fmt_clock(ts: float) -> str:
    """'10:45 PM' in the owner's time zone (config.timezone, see timers._dt)."""
    from .timers import _dt
    return _dt(ts).strftime("%I:%M %p").lstrip("0")


def fmt_hms(ts: float) -> str:
    from .timers import _dt
    return _dt(ts).strftime("%H:%M")


def _join(a: str, b: str) -> str:
    """Join text pieces: clauses get a space; fragments that carry their own spacing are joined as they are."""
    if not a:
        return b.strip()
    if not b:
        return a
    if a[-1].isspace() or b[0].isspace():
        return a + b
    return a + " " + b


# ----------------------------------------------------------------------------- conversation log ---
class ConversationLog:
    """The last MAX_TURNS exchanges: what the robot heard, what it said, how long it took to start answering.

    Local brain: heard_utterance() (one whole utterance, always starts a turn), said() once per clause, turn_done().
    Gemini Live: heard_fragment() / said() stream in pieces, turn_done() on turn_complete.
    """

    def __init__(self, backend: Callable[[], str] = lambda: "", clock: Callable[[], float] = time.time,
                 fmt: Callable[[float], str] | None = None, max_turns: int = MAX_TURNS, gap_s: float = GAP_S,
                 late_s: float = LATE_S):
        self._backend = backend
        self._clock = clock
        self._fmt = fmt or fmt_hms
        self._gap = gap_s
        self._late = late_s
        self._turns: collections.deque = collections.deque(maxlen=max_turns)
        self._cur: dict | None = None
        self._lock = threading.Lock()

    def _new(self, now: float) -> dict:
        t = {"ts": now, "heard": "", "said": "", "backend": self._backend() or "", "latency_s": None, "open": True,
             "_heard_at": now, "_closed_at": None}
        self._turns.append(t)
        self._cur = t
        return t

    def _close(self, now: float) -> None:
        if self._cur is not None:
            self._cur["open"] = False
            self._cur["_closed_at"] = now
            self._cur = None

    def heard_utterance(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            return
        with self._lock:
            now = self._clock()
            self._close(now)
            t = self._new(now)
            t["heard"] = text[:MAX_TEXT]

    def heard_fragment(self, text: str) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            now = self._clock()
            t = self._cur
            if t is not None and (t["said"] or now - t["_heard_at"] > self._gap and t["heard"] and not t["said"]):
                self._close(now)
                t = None
            if t is None:
                t = self._new(now)
            t["heard"] = _join(t["heard"], text)[:MAX_TEXT]
            t["_heard_at"] = now

    def said(self, text: str) -> None:
        if not text or not text.strip():
            return
        with self._lock:
            now = self._clock()
            t = self._cur
            if t is None and self._turns:
                last = self._turns[-1]
                if last["_closed_at"] is not None and now - last["_closed_at"] <= self._late and last["said"]:
                    t = last                      # a late transcript piece of the turn that just ended
            if t is None:
                t = self._new(now)                # an announcement (timer, greeting): nobody asked for it
                t["_heard_at"] = now
            if not t["said"] and t["latency_s"] is None and t["heard"]:
                t["latency_s"] = round(max(0.0, now - t["_heard_at"]), 1)
            t["said"] = _join(t["said"], text)[:MAX_TEXT]

    def turn_done(self) -> None:
        with self._lock:
            self._close(self._clock())

    def listing(self) -> list[dict]:
        """Newest first, for /api/state: {t, heard, said, backend, latency_s, open}."""
        with self._lock:
            return [{"t": self._fmt(t["ts"]), "heard": t["heard"], "said": t["said"], "backend": t["backend"],
                     "latency_s": t["latency_s"], "open": t["open"]} for t in reversed(self._turns)]

    def last_latency(self, backend: str | None = None) -> float | None:
        with self._lock:
            for t in reversed(self._turns):
                if t["latency_s"] is not None and (backend is None or t["backend"] == backend):
                    return t["latency_s"]
        return None


# ----------------------------------------------------------------------------- brain status chip ---
def brain_status(voice_status: dict, brain_state: str, last_latency_s: float | None) -> dict:
    """Extend VoiceBackend.status() with what the controller page needs for its "Brain:" chip.

    Adds state (the local brain's own state), last_latency_s and chip = {text, level} (level: ok / warn / bad).
    """
    st = dict(voice_status or {})
    active = st.get("active", "none")
    connected = bool(st.get("connected"))
    mode = st.get("mode", "auto")
    if active == "local":
        if not connected or brain_state in ("connecting", "starting", "error", "disconnected", "stopped"):
            text, level = "Brain: connecting…", "warn"
        elif brain_state == "loading":
            text, level = "Brain: warming up…", "warn"
        elif brain_state == "thinking":
            text, level = "Brain: thinking", "ok"
        else:
            text, level = "Brain: ready", "ok"
    elif active == "gemini":
        if mode == "gemini":
            text, level = "Brain: not used (Gemini only)", "ok"
        else:
            text, level = "Brain: offline — using Gemini", "warn"
    elif active == "none" and st.get("reason"):
        text, level = "Brain: offline", "bad"
    else:
        text, level = "Brain: starting…", "warn"
    if last_latency_s is not None and level == "ok":
        text += f" · last answer {last_latency_s:.1f} s"
    st.update({"state": brain_state, "last_latency_s": last_latency_s, "chip": {"text": text, "level": level}})
    return st


# ----------------------------------------------------------------------------- quiet mode ---
def clamp_minutes(v: Any) -> int:
    try:
        n = int(float(v))
    except (TypeError, ValueError):
        n = QUIET_DEFAULT_MIN
    return max(1, min(QUIET_MAX_MIN, n))


class QuietMode:
    """Mutes the mic and rests the head for a while, then wakes up on its own (or on talk / the page's Wake button).

    Timers and alarms are not touched: they ring through the voice backend, which quiet mode does not mute.
    """

    def __init__(self, set_muted: Callable[[bool], None], say: Callable[[str], None] | None = None, motion=None,
                 on_event: Callable[[str, str], None] | None = None, clock: Callable[[], float] = time.time,
                 fmt: Callable[[float], str] | None = None):
        self.set_muted = set_muted
        self.say = say or (lambda t: None)
        self.motion = motion
        self.on_event = on_event or (lambda k, t: None)
        self.clock = clock
        self.fmt = fmt or fmt_clock
        self.until = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def active(self) -> bool:
        return self.until > 0.0

    def enter(self, minutes: Any = QUIET_DEFAULT_MIN) -> dict:
        n = clamp_minutes(minutes)
        with self._lock:
            was = self.active
            self.until = self.clock() + n * 60
            until = self.until
        self.set_muted(True)
        if not was:
            self._rest()
        when = self.fmt(until)
        self.on_event("quiet", f"Quiet until {when}")
        unit = "minute" if n == 1 else "minutes"
        return {"say": f"Okay, I'll stay quiet for {n} {unit}. Press talk to wake me.", "until": when, "minutes": n}

    def wake(self, say: bool = True, reason: str = "") -> bool:
        with self._lock:
            if not self.active:
                return False
            self.until = 0.0
        self.set_muted(False)
        self._stand_up()
        self.on_event("quiet", "Quiet mode over" + (f" ({reason})" if reason else ""))
        if say:
            try:
                self.say("I'm back.")
            except Exception:
                log.exception("quiet wake announcement failed")
        return True

    def on_talk(self) -> None:
        """Hook for app.talk_pressed_hooks."""
        self.wake(reason="talk pressed")

    def tick(self) -> None:
        if self.active and self.clock() >= self.until:
            self.wake(reason="time is up")

    def status(self) -> dict:
        if not self.active:
            return {"active": False, "until": 0, "until_text": "", "minutes_left": 0}
        left = max(0.0, self.until - self.clock())
        return {"active": True, "until": self.until, "until_text": self.fmt(self.until),
                "minutes_left": int(left // 60) + (1 if left % 60 else 0)}

    def _rest(self) -> None:
        m = self.motion
        if m is None:
            return
        try:
            m.emotion("sleepy")
        except Exception as e:
            log.info("quiet mode: could not rest the head (%s)", e)

    def _stand_up(self) -> None:
        m = self.motion
        if m is None:
            return
        try:
            m.move_head("front")
        except Exception as e:
            log.info("quiet mode: could not lift the head (%s)", e)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()

        def run() -> None:
            while not self._stop.wait(1.0):
                try:
                    self.tick()
                except Exception:
                    log.exception("quiet mode tick failed")

        self._thread = threading.Thread(target=run, daemon=True, name="quiet-mode")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()


# ----------------------------------------------------------------------------- "I'm ready" greeting ---
class Greeter:
    """Says a short hello the first time the voice backend is ready after the app starts (once per app start)."""

    def __init__(self, say: Callable[[str], None], enabled: Callable[[], bool] = lambda: True,
                 is_quiet: Callable[[], bool] = lambda: False, text: str = "I'm ready."):
        self.say = say
        self.enabled = enabled
        self.is_quiet = is_quiet
        self.text = text
        self._done = False
        self._lock = threading.Lock()

    def ready(self, source: str = "") -> bool:
        """Call whenever a backend becomes ready (every reconnect); only the first call can speak."""
        with self._lock:
            if self._done:
                return False
            self._done = True
        if not self.enabled() or self.is_quiet():
            return False
        try:
            self.say(self.text)
        except Exception:
            log.exception("greeting failed")
            return False
        return True
