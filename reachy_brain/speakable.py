"""Turn digits into words before text goes to the TTS: Pocket TTS reads "254" or "15%" badly, and Gemini's voice
is the one people compare it with. Pure functions, no dependencies."""
from __future__ import annotations

import re

_ONES = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
         "seventeen eighteen nineteen").split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_ORD = {"one": "first", "two": "second", "three": "third", "five": "fifth", "eight": "eighth", "nine": "ninth",
        "twelve": "twelfth"}


def _below_100(n: int) -> str:
    if n < 20:
        return _ONES[n]
    return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])


def _below_1000(n: int) -> str:
    h, r = divmod(n, 100)
    return " ".join(x for x in ((_ONES[h] + " hundred") if h else "", _below_100(r) if r else "") if x)


def int_words(n: int) -> str:
    if n < 0:
        return "minus " + int_words(-n)
    if n < 100:
        return _below_100(n)
    parts = []
    for size, name in ((10**9, "billion"), (10**6, "million"), (10**3, "thousand")):
        q, n = divmod(n, size)
        if q:
            parts.append(_below_1000(q) + " " + name)
    if n:
        parts.append(_below_1000(n))
    return " ".join(parts)


def _pairs(n: int) -> str:
    """Read 1100-9999 the way people say years and FRC team numbers: 1678 -> sixteen seventy eight."""
    a, b = divmod(n, 100)
    if b == 0:
        return int_words(a) + " hundred"
    return _below_100(a) + " " + ("oh " + _ONES[b] if b < 10 else _below_100(b))


def _number(tok: str) -> str:
    n = int(tok)
    if len(tok) == 3 and n >= 100 and tok[0] != "0" and n % 100 != 0:      # team 254 -> two fifty four
        return _ONES[int(tok[0])] + " " + (("oh " + _ONES[int(tok[1:])]) if tok[1] == "0" else _below_100(int(tok[1:])))
    if len(tok) == 4 and 1001 <= n <= 9999 and n % 1000 != 0:
        return _pairs(n)
    return int_words(n)


def _ordinal(n: int) -> str:
    w = int_words(n)
    head, _, last = w.rpartition(" ")
    last = last or head
    base = _ORD.get(last) or (last[:-1] + "ieth" if last.endswith("y") else last + "th")
    return (head + " " if _ and head else "") + base


def speakable(text: str) -> str:
    t = text
    t = re.sub(r"(\d{1,2}):(\d{2})\s*([ap])\.?m\.?", lambda m: _clock(m.group(1), m.group(2)) + (" a m" if m.group(3).lower() == "a" else " p m"), t, flags=re.I)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b", lambda m: _clock(m.group(1), m.group(2)), t)
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b", lambda m: _ordinal(int(m.group(1))), t)
    t = re.sub(r"(\d+)\s*%", lambda m: m.group(1) + " percent", t)
    t = re.sub(r"\b(\d{1,3}(?:,\d{3})+)\b", lambda m: int_words(int(m.group(1).replace(",", ""))), t)
    t = re.sub(r"(\d+)\.(\d+)", lambda m: int_words(int(m.group(1))) + " point " + " ".join(_ONES[int(d)] for d in m.group(2)), t)
    t = re.sub(r"\d+", lambda m: _number(m.group(0)), t)
    return t


def _clock(h: str, m: str) -> str:
    hh, mm = int(h), int(m)
    return int_words(hh) + (" o'clock" if mm == 0 else (" oh " + _ONES[mm] if mm < 10 else " " + _below_100(mm)))
