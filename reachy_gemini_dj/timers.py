"""Timers, reminders and alarms that run on the robot.

The brain passes free text ("an hour and a half", "7 tonight"); everything numeric is parsed here.
Every tool result is a dict with a short, complete, speakable ``say`` string (the local brain speaks it
as is), or ``{"error": "..."}`` with a speakable message.

Fire times are absolute and saved to disk, so timers survive an app restart. Ones that expired while the
app was off are dropped, except those that went off in the last 10 minutes, which are announced as missed.
"""
from __future__ import annotations

import datetime
import json
import logging
import math
import os
import re
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

log = logging.getLogger(__name__)

# Time zone for clock times ("wake me at 7:30", "at 3:45 PM"). The robot ships set to Europe/London, so the app sets
# this from the controller page's browser time zone (config.timezone). None = the robot's own system time zone.
_TZ = None


def set_timezone(name: str) -> bool:
    """Use this IANA time zone (e.g. "America/Chicago") for clock times; "" = system zone. False if unknown."""
    global _TZ
    if not name:
        _TZ = None
        return True
    try:
        from zoneinfo import ZoneInfo
        _TZ = ZoneInfo(name)
        return True
    except Exception:
        return False


def _dt(ts: float) -> datetime.datetime:
    return datetime.datetime.fromtimestamp(ts, _TZ)

DEFAULT_PATH = Path.home() / ".reachy_gemini_dj" / "timers.json"
MAX_DURATION_S = 24 * 3600
MAX_ACTIVE = 20
MISSED_WINDOW_S = 600
REPEAT_S = 20.0
MAX_RINGS = 3
TICK_S = 0.5
MISSED_DELAY_S = 15.0       # give the voice backend time to connect before announcing missed timers
SPEECH_GRACE_S = 2.5        # speech this soon after an announcement is probably our own echo
DUCK_LEVEL = 0.25
CHIME_SR = 24000            # AudioPipeline.play_voice takes 24 kHz 16-bit mono PCM


# ------------------------------------------------------------------------------------ parsing ---
_ONES = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
         "eighty": 80, "ninety": 90}
_UNITS = {}
for _names, _secs in ((("s", "sec", "secs", "second", "seconds"), 1),
                      (("m", "min", "mins", "minute", "minutes"), 60),
                      (("h", "hr", "hrs", "hour", "hours"), 3600),
                      (("d", "day", "days"), 86400)):
    for _n in _names:
        _UNITS[_n] = _secs
_DUR_SKIP = {"for", "in", "about", "around", "roughly", "approximately", "timer", "set", "me", "please", "after",
             "from", "now", "within", "the"}
_FRACS = {"half": 0.5, "quarter": 0.25}


def _tokens(text: str, pattern: str) -> list[str]:
    return re.findall(pattern, str(text).lower())


def _word_num(toks: list[str], i: int) -> tuple[int, int] | None:
    """Read a spelled number ("forty five", "one hundred twenty") starting at toks[i]."""
    n = len(toks)
    if i >= n:
        return None
    w = toks[i]
    if w in _ONES:
        v = _ONES[w]
        i += 1
    elif w in _TENS:
        v = _TENS[w]
        i += 1
        if i < n and toks[i] in _ONES and 1 <= _ONES[toks[i]] <= 9:
            v += _ONES[toks[i]]
            i += 1
        return v, i
    else:
        return None
    if i < n and toks[i] == "hundred":
        v *= 100
        i += 1
        if i < n and toks[i] in _TENS:
            v += _TENS[toks[i]]
            i += 1
            if i < n and toks[i] in _ONES and 1 <= _ONES[toks[i]] <= 9:
                v += _ONES[toks[i]]
                i += 1
        elif i < n and toks[i] in _ONES and 0 < _ONES[toks[i]] < 20:
            v += _ONES[toks[i]]
            i += 1
    return v, i


def _qty(toks: list[str], i: int) -> tuple[float, int] | None:
    """A quantity: digits, a spelled number, an article, or a couple."""
    n = len(toks)
    if i >= n:
        return None
    t = toks[i]
    if re.fullmatch(r"\d+(?:\.\d+)?", t):
        return float(t), i + 1
    if t in ("a", "an") and i + 1 < n and toks[i + 1] == "couple":
        i += 1
        t = "couple"
    if t == "couple":
        i += 1
        if i < n and toks[i] == "of":
            i += 1
        return 2.0, i
    if t in ("a", "an"):
        return 1.0, i + 1
    r = _word_num(toks, i)
    return (float(r[0]), r[1]) if r else None


def _and_frac(toks: list[str], i: int) -> tuple[float, int] | None:
    """Match "and a half" / "and half" / "and a quarter" at toks[i]; returns (fraction, next index)."""
    n = len(toks)
    if i < n and toks[i] == "and":
        j = i + 1
        if j < n and toks[j] in ("a", "an"):
            j += 1
        if j < n and toks[j] in _FRACS:
            return _FRACS[toks[j]], j + 1
    return None


DURATION_HINT = "Try something like 10 minutes, or an hour and a half."


def parse_duration(text: Any) -> int:
    """Free text duration to whole seconds. Raises ValueError with a speakable message."""
    bad = ValueError(f"Sorry, I didn't understand how long that is. {DURATION_HINT}")
    toks = _tokens(text, r"\d+(?:\.\d+)?|[a-z]+|:")
    toks = [t for t in toks if t not in _DUR_SKIP]
    # a stray article ("a timer for 10 minutes") is dropped; "a minute", "a half hour", "a couple" keep theirs
    toks = [t for k, t in enumerate(toks) if not (t in ("a", "an") and not (
        k + 1 < len(toks) and (toks[k + 1] in _UNITS or toks[k + 1] in ("quarter", "couple"))))
            and not (t in ("a", "an") and k + 1 < len(toks) and toks[k + 1] == "half"
                     and k + 2 < len(toks) and toks[k + 2] in _UNITS)]
    if not toks:
        raise bad
    n, i, total, parts = len(toks), 0, 0.0, 0
    while i < n:
        if toks[i] == "and":
            i += 1
            continue
        t = toks[i]
        if t in _FRACS:                                  # "half an hour", "a quarter of an hour"
            qty = _FRACS[t]
            i += 1
            while i < n and toks[i] in ("of", "a", "an"):
                i += 1
        else:
            r = _qty(toks, i)
            if r is None:
                raise bad
            qty, i = r
            if i < n and toks[i] in ("quarter", "quarters"):       # "three quarters of an hour"
                qty *= 0.25
                i += 1
                while i < n and toks[i] in ("of", "a", "an"):
                    i += 1
            else:
                af = _and_frac(toks, i)                            # "one and a half minutes"
                if af and af[1] < n and toks[af[1]] in _UNITS:
                    qty += af[0]
                    i = af[1]
        if i >= n or toks[i] not in _UNITS:
            if parts == 0 and i >= n and t not in _FRACS:
                total, parts = qty * 60, 1               # a bare number means minutes
                break
            raise bad
        unit = _UNITS[toks[i]]
        i += 1
        total += qty * unit
        parts += 1
        af = _and_frac(toks, i)                          # "an hour and a half"
        if af and (af[1] >= n):
            total += af[0] * unit
            i = af[1]
    secs = int(round(total))
    if secs <= 0:
        raise ValueError("That isn't a length of time I can use. Try 10 minutes, for example.")
    if secs > MAX_DURATION_S:
        raise ValueError("I can only set timers up to 24 hours. Use an alarm for something later.")
    return secs


_PERIOD = {"morning": "am", "afternoon": "pm", "evening": "pm", "tonight": "pm", "night": "night"}
_CLOCK_SKIP = {"at", "in", "the", "on", "for", "by", "around", "about", "sharp", "exactly", "me", "my", "wake",
               "up", "alarm", "set", "please", "oclock", "clock", "is", "be", "this", "today", "a", "an", "of",
               "to", "o", "after", "past", "till", "til", "before", "and"}
CLOCK_HINT = "Try something like 7:30 AM or 6 tonight."


def _fmt_clock(h: int, m: int) -> str:
    if (h, m) == (12, 0):
        return "noon"
    if (h, m) == (0, 0):
        return "midnight"
    suffix = "AM" if h < 12 else "PM"
    h12 = h % 12 or 12
    return f"{h12} {suffix}" if m == 0 else f"{h12}:{m:02d} {suffix}"


def parse_clock(text: Any, now: float) -> tuple[float, int, int]:
    """Free text clock time to (epoch seconds of the next occurrence, hour, minute). Local time.

    Raises ValueError with a speakable message."""
    bad = ValueError(f"Sorry, I didn't understand that time. {CLOCK_HINT}")
    s = re.sub(r"\b([ap])\.?\s?m\.?(?![a-z])", r"\1m", str(text).lower())
    toks = _tokens(s, r"\d+[:.]\d{2}|\d+|[a-z]+")
    ampm, period, period_word, day = None, None, "", 0
    items: list[tuple] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if re.fullmatch(r"\d+[:.]\d{2}", t):
            h, m = (int(x) for x in re.split(r"[:.]", t))
            items.append(("HM", h, m))
        elif t.isdigit():
            items.append(("NUM", int(t)))
        elif t in ("am", "pm"):
            ampm = t
        elif t in ("noon", "midday"):
            items.append(("HM", 12, 0, True))
        elif t == "midnight":
            items.append(("HM", 0, 0, True))
        elif t == "tomorrow":
            day = 1
        elif t in _PERIOD:
            period, period_word = _PERIOD[t], t
        elif t in _FRACS:
            items.append(("FRAC", 30 if t == "half" else 15))
        elif t == "oh" or t in _ONES or t in _TENS:
            if t == "oh":
                items.append(("NUM", 0))
            else:
                r = _word_num(toks, i)
                items.append(("NUM", r[0]))
                i = r[1] - 1
        elif t in ("past", "after", "to", "till", "til", "before"):
            items.append(("REL", "past" if t in ("past", "after") else "to"))
        elif t in _CLOCK_SKIP:
            pass
        else:
            raise bad
        i += 1
    # a stray "to"/"past" (as in "set an alarm to 7") is dropped unless it joins two quantities
    cleaned: list[tuple] = []
    for k, it in enumerate(items):
        if it[0] == "REL":
            ok = (0 < k < len(items) - 1 and items[k - 1][0] in ("NUM", "FRAC") and items[k + 1][0] == "NUM")
            if not ok:
                continue
        cleaned.append(it)
    items = cleaned
    definite = False
    kinds = [it[0] for it in items]
    if kinds == ["HM"]:
        h, m = items[0][1], items[0][2]
        definite = len(items[0]) > 3
    elif kinds == ["NUM"]:
        v = items[0][1]
        h, m = (v // 100, v % 100) if v >= 100 else (v, 0)
    elif kinds == ["NUM", "NUM"]:
        h, m = items[0][1], items[1][1]
    elif kinds == ["NUM", "NUM", "NUM"] and items[1][1] == 0 and items[2][1] < 10:     # "seven oh five"
        h, m = items[0][1], items[2][1]
    elif kinds in (["NUM", "REL", "NUM"], ["FRAC", "REL", "NUM"]):
        q, h = items[0][1], items[2][1]
        if items[1][1] == "past":
            m = q
        else:
            m = 60 - q
            h = (h - 1) if h > 1 else 12
    else:
        raise bad
    if m >= 60 or h < 0 or h > 24 or (h == 24 and m):
        raise bad
    if h == 24:
        h = 0
    # which hours can this mean?
    if definite or h == 0 or h > 12:
        cands = [h]
    else:
        if ampm == "am" or period == "am":
            cands = [h % 12]
        elif ampm == "pm":
            cands = [h % 12 + 12]
        elif period == "pm":              # afternoon / evening / tonight ("12 tonight" is midnight)
            cands = [0] if h == 12 and period_word in ("evening", "tonight") else [h % 12 + 12]
        elif period == "night":
            cands = [0 if h == 12 else (h + 12 if h >= 6 else h)]
        else:
            cands = [h % 12, h % 12 + 12]          # "7:30": whichever comes next
    start = _dt(now)
    best = None
    for ch in cands:
        base = start.replace(hour=ch, minute=m, second=0, microsecond=0)
        if day:
            base += datetime.timedelta(days=1)
        elif base.timestamp() <= now + 1:
            base += datetime.timedelta(days=1)
        if best is None or base < best:
            best = base
    return best.timestamp(), best.hour, best.minute


# -------------------------------------------------------------------------------- formatting ---
_COUNT = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def fmt_duration(secs: float) -> str:
    """Exact: 600 -> '10 minutes', 5400 -> '1 hour and 30 minutes', 90 -> '1 minute and 30 seconds'."""
    secs = int(round(secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    parts = [p for p in (_plural(h, "hour") if h else "", _plural(m, "minute") if m else "",
                         _plural(s, "second") if s else "") if p]
    return " and ".join(parts) if len(parts) <= 2 else ", ".join(parts[:-1]) + " and " + parts[-1]


def fmt_left(secs: float) -> str:
    """Time remaining, rounded the way a person would say it."""
    secs = max(0, math.ceil(secs))
    if secs >= 3600:
        m = round((secs - secs // 3600 * 3600) / 60)
        h = secs // 3600 + (m == 60)
        m = 0 if m == 60 else m
        return fmt_duration(h * 3600 + m * 60)
    if secs >= 120:
        return _plural(math.ceil(secs / 60), "minute")
    return fmt_duration(secs)


_REMINDER_VERBS = {"check", "call", "take", "turn", "get", "buy", "water", "feed", "send", "pick", "put", "bring",
                   "go", "make", "pay", "email", "text", "drink", "stretch", "stand", "leave", "start", "stop",
                   "wake", "clean", "cook", "remember", "walk", "charge", "plug", "unplug", "close", "open", "do",
                   "meet", "submit", "finish", "read", "pack", "lock", "move", "eat", "play", "join", "return"}


def _clean_label(label: Any) -> tuple[str, bool]:
    """(label text, is_reminder). 'to check the battery' / 'check the battery' are reminders; 'pizza' names a timer."""
    s = re.sub(r"\s+", " ", str(label or "")).strip().strip(".,!?;:").strip()
    low = s.lower()
    for pre in ("remind me to ", "to ", "that i should ", "that i need to ", "that "):
        if low.startswith(pre):
            return s[len(pre):].strip(), True
    words = low.split()
    if words and (words[0] in _REMINDER_VERBS or len(words) >= 3):
        return s, True
    return s, False


def _day_word(fire_at: float, now: float) -> str:
    d1 = _dt(fire_at).date()
    d0 = _dt(now).date()
    delta = (d1 - d0).days
    return " tomorrow" if delta == 1 else "" if delta == 0 else " on " + d1.strftime("%A")


def _make_chime() -> bytes:
    """A short two-note chime as 24 kHz 16-bit mono PCM."""
    out = bytearray()
    for freq, dur in ((880.0, 0.16), (1318.5, 0.38)):
        n = int(CHIME_SR * dur)
        for k in range(n):
            t = k / CHIME_SR
            env = min(1.0, k / 120) * math.exp(-4.0 * t / dur)
            v = 0.5 * math.sin(2 * math.pi * freq * t) + 0.2 * math.sin(2 * math.pi * freq * 2 * t)
            out += struct.pack("<h", int(max(-1.0, min(1.0, v * env)) * 0.45 * 32767))
    return bytes(out)


# ------------------------------------------------------------------------------------- class ---
class Timers:
    def __init__(self, say: Callable[[str], None] | None = None, on_event: Callable[[str, str], None] | None = None,
                 motion=None, music=None, clock: Callable[[], float] = time.time, path: str | os.PathLike | None = None,
                 play_pcm: Callable[[bytes], None] | None = None,
                 user_speech_age: Callable[[], float] | None = None,
                 repeat_s: float = REPEAT_S, max_rings: int = MAX_RINGS, missed_delay_s: float = MISSED_DELAY_S):
        """user_speech_age: seconds since the user last spoke; polled so speech dismisses a ringing alarm."""
        self.say = say or (lambda t: None)
        self.on_event = on_event or (lambda k, t: None)
        self.motion = motion
        self.music = music
        self.clock = clock
        self.path = Path(path) if path else DEFAULT_PATH
        self.play_pcm = play_pcm
        self.user_speech_age = user_speech_age
        self.repeat_s = repeat_s
        self.max_rings = max_rings
        self.missed_delay_s = missed_delay_s
        self._lock = threading.RLock()
        self._items: list[dict] = []          # waiting
        self._ringing: list[dict] = []
        self._missed: list[dict] = []
        self._missed_at = 0.0
        self._next_id = 1
        self._ducked = False
        self._last_dismiss = -1e9
        self._chime: bytes | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._load()

    # ---------------- persistence ----------------
    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except Exception as e:
            log.warning("could not read %s: %s", self.path, e)
            return
        now = self.clock()
        self._next_id = int(data.get("next_id", 1) or 1)
        for e in data.get("timers", []):
            try:
                e = {"id": int(e["id"]), "kind": str(e["kind"]), "fire_at": float(e["fire_at"]),
                     "created_at": float(e.get("created_at", now)), "duration": e.get("duration"),
                     "label": str(e.get("label", "")), "reminder": bool(e.get("reminder", False))}
            except Exception:
                continue
            if e["fire_at"] > now:
                self._items.append(e)
            elif now - e["fire_at"] <= MISSED_WINDOW_S:
                self._missed.append(e)
            self._next_id = max(self._next_id, e["id"] + 1)
        if self._missed:
            self._save()          # missed ones are announced once, then forgotten

    def _save(self) -> None:
        with self._lock:
            data = {"version": 1, "next_id": self._next_id,
                    "timers": [{k: e[k] for k in ("id", "kind", "fire_at", "created_at", "duration", "label", "reminder")}
                               for e in self._items]}
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data), encoding="utf-8")
            os.replace(tmp, self.path)
        except Exception as e:
            log.warning("could not save timers: %s", e)

    # ---------------- descriptions ----------------
    def _clock_text(self, e: dict, now: float | None = None) -> str:
        dt = _dt(e["fire_at"])
        txt = _fmt_clock(dt.hour, dt.minute)
        return txt + (_day_word(e["fire_at"], now if now is not None else self.clock()))

    def _describe_waiting(self, e: dict, now: float) -> str:
        """For list_timers: '4 minutes left on the pizza timer', 'an alarm at 7:30 AM', 'a reminder to ... at ...'"""
        if e["reminder"] and e["kind"] == "alarm":
            return f"a reminder to {e['label']} at {self._clock_text(e, now)}"
        if e["kind"] == "alarm":
            return f"an alarm at {self._clock_text(e, now)}" + (f" called {e['label']}" if e["label"] else "")
        left = fmt_left(e["fire_at"] - now)
        if e["reminder"]:
            return f"a reminder to {e['label']} in {left}"
        if e["label"]:
            return f"{left} left on the {e['label']} timer"
        return f"{left} left on a timer"

    def _new_id(self) -> int:
        i = self._next_id
        self._next_id += 1
        return i

    def _add(self, kind: str, fire_at: float, duration: float | None, label: str, reminder: bool, now: float) -> dict:
        e = {"id": self._new_id(), "kind": kind, "fire_at": fire_at, "created_at": now, "duration": duration,
             "label": label, "reminder": reminder}
        self._items.append(e)
        self._save()
        return e

    # ---------------- tools ----------------
    def set_timer(self, duration: str = "", label: str = "") -> dict:
        try:
            secs = parse_duration(duration)
        except ValueError as err:
            return {"error": str(err)}
        with self._lock:
            if len(self._items) >= MAX_ACTIVE:
                return {"error": "You already have too many timers. Cancel one first."}
            now = self.clock()
            text, reminder = _clean_label(label)
            e = self._add("timer", now + secs, secs, text, reminder, now)
        if reminder:
            say = f"Okay, I'll remind you to {text} at {self._clock_text(e, now)}."
        elif text:
            say = f"Timer set for {fmt_duration(secs)}, called {text}."
        else:
            say = f"Timer set for {fmt_duration(secs)}."
        self.on_event("timer", say)
        return {"ok": True, "id": e["id"], "say": say}

    def set_alarm(self, time: str = "", label: str = "") -> dict:      # noqa: A002  (tool arg name)
        now = self.clock()
        try:
            fire_at, h, m = parse_clock(time, now)
        except ValueError as err:
            return {"error": str(err)}
        with self._lock:
            if len(self._items) >= MAX_ACTIVE:
                return {"error": "You already have too many timers. Cancel one first."}
            text, reminder = _clean_label(label)
            e = self._add("alarm", fire_at, None, text, reminder, now)
        when = self._clock_text(e, now)
        if reminder:
            say = f"Okay, I'll remind you to {text} at {when}."
        elif text:
            say = f"Alarm set for {when}, called {text}."
        else:
            say = f"Alarm set for {when}."
        self.on_event("timer", say)
        return {"ok": True, "id": e["id"], "say": say}

    def list_timers(self) -> dict:
        with self._lock:
            now = self.clock()
            items = sorted(self._items, key=lambda e: e["fire_at"])
            ringing = list(self._ringing)
        if not items:
            say = "Something is ringing right now. " if ringing else ""
            return {"timers": [], "say": say + "You have no timers or alarms."}
        descs = [self._describe_waiting(e, now) for e in items]
        n = len(items)
        noun = "alarm" if all(e["kind"] == "alarm" for e in items) else "timer"
        if n == 1:
            say = f"You have one {noun}: {descs[0]}."
        else:
            body = f"{descs[0]}, and {descs[1]}" if n == 2 else ", ".join(descs[:-1]) + ", and " + descs[-1]
            say = f"You have {_COUNT.get(n, n)} {noun}s: {body}."
        return {"timers": self.status()["timers"], "say": say}

    def _match(self, which: str) -> list[dict]:
        """Items matching free text: a label, a duration ("10 minutes") or a clock time ("7:30")."""
        items = list(self._items)
        low = re.sub(r"\b(the|my|a|an|timer|timers|alarm|alarms|reminder|reminders|for|called|named)\b", " ", which.lower())
        low = re.sub(r"\s+", " ", low).strip()
        hits = [e for e in items if low and e["label"] and (low in e["label"].lower() or e["label"].lower() in low)]
        if hits:
            return hits
        try:
            secs = parse_duration(low or which)
            hits = [e for e in items if e["kind"] == "timer" and e["duration"] and abs(e["duration"] - secs) < 1]
            if hits:
                return hits
        except ValueError:
            pass
        try:
            _, h, m = parse_clock(which, self.clock())
            hits = [e for e in items if _dt(e["fire_at"]).timetuple()[3:5] == (h, m)]
        except ValueError:
            hits = []
        if not hits and re.search(r"\balarms?\b", which.lower()) and not low:
            hits = [e for e in items if e["kind"] == "alarm"]
        return hits

    def cancel_timer(self, which: str = "") -> dict:
        which = str(which or "").strip()
        low = which.lower()
        with self._lock:
            if not which:
                if not self._items:
                    if self._ringing:
                        return self.stop_alarm()
                    return {"error": "You have no timers or alarms to cancel."}
                victims = [max(self._items, key=lambda e: e["created_at"] * 1e6 + e["id"])]
            elif re.fullmatch(r"(all|everything|all of them|all of (the|my) (timers|alarms)|"
                              r"(all|every)( my| the)? (timers|alarms|timers and alarms|alarms and timers))", low):
                if "alarm" in low and "timer" not in low:
                    victims = [e for e in self._items if e["kind"] == "alarm"]
                elif "timer" in low and "alarm" not in low:
                    victims = [e for e in self._items if e["kind"] == "timer"]
                else:
                    victims = list(self._items)
                if not victims:
                    return {"error": "You have no timers or alarms to cancel."}
            else:
                victims = self._match(which)
                if not victims:
                    if not self._items:
                        return {"error": "You have no timers or alarms to cancel."}
                    return {"error": f"I couldn't find a timer or alarm matching {which}."}
            ids = {e["id"] for e in victims}
            self._items = [e for e in self._items if e["id"] not in ids]
            self._save()
        if len(victims) == 1:
            say = f"Okay, cancelled {self._cancel_phrase(victims[0])}."
        else:
            kinds = {v["kind"] for v in victims}
            noun = f"{victims[0]['kind']}s" if len(kinds) == 1 else "timers and alarms"
            n = len(victims)
            say = ("Okay, cancelled the timer and the alarm." if n == 2 and len(kinds) > 1 else
                   f"Okay, cancelled both {noun}." if n == 2 else
                   f"Okay, cancelled all {_COUNT.get(n, n)} {noun}.")
        self.on_event("timer", say)
        return {"ok": True, "cancelled": len(victims), "say": say}

    def _cancel_phrase(self, e: dict) -> str:
        if e["reminder"]:
            return f"the reminder to {e['label']}"
        if e["kind"] == "alarm":
            dt = _dt(e["fire_at"])
            return f"the {_fmt_clock(dt.hour, dt.minute)} alarm" if not e["label"] else f"the {e['label']} alarm"
        if e["label"]:
            return f"the {e['label']} timer"
        return f"the {fmt_duration(e['duration'])} timer" if e["duration"] else "the timer"

    def cancel_id(self, tid: int) -> dict:
        with self._lock:
            hit = [e for e in self._items if e["id"] == tid]
            if not hit:
                return {"error": "That timer is already gone."}
            self._items = [e for e in self._items if e["id"] != tid]
            self._save()
        return {"ok": True, "say": f"Okay, cancelled {self._cancel_phrase(hit[0])}."}

    def stop_alarm(self) -> dict:
        n = self.dismiss()
        if n:
            return {"ok": True, "say": "Okay, stopped."}
        if self.clock() - self._last_dismiss < 30:
            return {"ok": True, "say": "Okay."}
        return {"ok": True, "say": "Nothing is ringing right now."}

    def handle(self, name: str, args: dict) -> dict:
        args = args or {}
        if name == "set_timer":
            return self.set_timer(args.get("duration", ""), args.get("label", ""))
        if name == "set_alarm":
            return self.set_alarm(args.get("time", ""), args.get("label", ""))
        if name == "list_timers":
            return self.list_timers()
        if name == "cancel_timer":
            return self.cancel_timer(args.get("which", ""))
        if name == "stop_alarm":
            return self.stop_alarm()
        return {"error": f"Unknown function: {name}"}

    # ---------------- state for the controller page ----------------
    def status(self) -> dict:
        with self._lock:
            now = self.clock()
            rows = [{"id": e["id"], "kind": e["kind"], "label": e["label"], "reminder": e["reminder"],
                     "fire_at": e["fire_at"], "left": max(0.0, e["fire_at"] - now), "ringing": False,
                     "at": self._clock_text(e, now)} for e in sorted(self._items, key=lambda e: e["fire_at"])]
            ring = [{"id": e["id"], "kind": e["kind"], "label": e["label"], "reminder": e["reminder"],
                     "fire_at": e["fire_at"], "left": 0.0, "ringing": True, "at": ""} for e in self._ringing]
            return {"timers": ring + rows, "ringing": bool(self._ringing)}

    # ---------------- ringing ----------------
    def dismiss(self) -> int:
        """Silence everything that is ringing. Returns how many were ringing."""
        with self._lock:
            n = len(self._ringing)
            self._ringing = []
            if n:
                self._last_dismiss = self.clock()
        if n:
            self._unduck()
            self.on_event("timer", "Alarm dismissed")
        return n

    def on_user_speech(self) -> None:
        self.dismiss()

    def _unduck(self) -> None:
        if self._ducked and self.music is not None:
            self._ducked = False
            try:
                self.music.reset_duck()
            except Exception as e:
                log.debug("unduck: %s", e)

    def _ring_text(self, e: dict, repeat: int) -> str:
        if e["reminder"]:
            return f"Reminder: {e['label']}."
        if e["kind"] == "alarm":
            dt = _dt(e["fire_at"])
            base = f"It's {_fmt_clock(dt.hour, dt.minute)}. This is your alarm."
            return f"It's {_fmt_clock(dt.hour, dt.minute)}. This is your {e['label']} alarm." if e["label"] else base
        if e["label"]:
            return f"Your {e['label']} timer is done."
        return f"Your {fmt_duration(e['duration'])} timer is done." if e["duration"] else "Your timer is done."

    def _announce(self, e: dict) -> None:
        text = self._ring_text(e, e["ring_n"])
        self.on_event("timer", text)
        if self.play_pcm is not None:
            try:
                if self._chime is None:
                    self._chime = _make_chime()
                self.play_pcm(self._chime)
            except Exception as err:
                log.debug("chime: %s", err)
        if self.motion is not None:
            try:
                self.motion.emotion("surprised")
            except Exception as err:
                log.debug("ring motion: %s", err)
        try:
            self.say(text)
        except Exception as err:
            log.warning("could not speak timer: %s", err)

    def tick(self) -> None:
        """One pass: fire due timers, repeat announcements, honour speech dismissal, announce missed ones."""
        now = self.clock()
        announce: list[dict] = []
        changed = False
        with self._lock:
            for e in [e for e in self._items if e["fire_at"] <= now]:
                self._items.remove(e)
                e.update(ring_n=0, next_ring=now, ann_t=now)
                self._ringing.append(e)
                changed = True
            # user spoke after the last announcement (and not just our own echo): dismiss those
            announce_dismiss = False
            if self._ringing and self.user_speech_age is not None:
                try:
                    spoke_at = now - float(self.user_speech_age())
                except Exception:
                    spoke_at = -1e18
                heard = [e for e in self._ringing if e["ring_n"] > 0 and spoke_at > e["ann_t"] + SPEECH_GRACE_S]
                if heard:
                    self._ringing = [e for e in self._ringing if e not in heard]
                    self._last_dismiss = now
                    announce_dismiss = True
            for e in list(self._ringing):
                if now >= e["next_ring"]:
                    if e["ring_n"] >= self.max_rings:
                        self._ringing.remove(e)
                        continue
                    e["ring_n"] += 1
                    e["ann_t"] = now
                    e["next_ring"] = now + self.repeat_s
                    announce.append(e)
            ringing_now = bool(self._ringing)
            missed = []
            if self._missed and now >= self._missed_at:
                missed, self._missed = self._missed, []
        if changed:
            self._save()
        if announce_dismiss:
            self._unduck()
            self.on_event("timer", "Alarm dismissed by voice")
        for e in announce:
            self._announce(e)
        if announce or (ringing_now and any(now - e["ann_t"] > 5.0 for e in self._ringing)):
            self._duck()           # also re-applied between announcements: the audio pipeline un-ducks after speech
        if not ringing_now:
            self._unduck()
        if missed:
            self._announce_missed(missed, now)

    def _duck(self) -> None:
        if self.music is None:
            return
        try:
            self.music.set_duck(DUCK_LEVEL)
            self._ducked = True
        except Exception as e:
            log.debug("duck: %s", e)

    def _announce_missed(self, missed: list[dict], now: float) -> None:
        for e in missed:
            ago = _plural(max(1, round((now - e["fire_at"]) / 60)), "minute")
            if e["reminder"]:
                what = f"your reminder to {e['label']}"
            elif e["kind"] == "alarm":
                what = "your alarm"
            else:
                what = f"your {e['label']} timer" if e["label"] else (
                    f"your {fmt_duration(e['duration'])} timer" if e["duration"] else "your timer")
            text = f"Sorry, I missed {what} while I was off. It was due {ago} ago."
            self.on_event("timer", text)
            try:
                self.say(text)
            except Exception as err:
                log.warning("could not speak missed timer: %s", err)

    # ---------------- thread ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._missed_at = self.clock() + self.missed_delay_s
        self._thread = threading.Thread(target=self._loop, daemon=True, name="timers")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._unduck()

    def _loop(self) -> None:
        while not self._stop.wait(TICK_S):
            try:
                self.tick()
            except Exception:
                log.exception("timer tick failed")
