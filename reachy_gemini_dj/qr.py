"""Read QR codes from the robot camera, on the robot (nothing is sent anywhere to decode)."""
from __future__ import annotations

import logging
import re
import time
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np

log = logging.getLogger(__name__)


def decode(frame) -> list[str]:
    """Return the text of every QR code in a BGR/RGB/gray image. Tries zxing-cpp, then OpenCV."""
    img = np.asarray(frame)
    try:
        import zxingcpp
        return [b.text for b in zxingcpp.read_barcodes(img) if b.text]
    except ImportError:
        pass
    try:
        import cv2
        ok, texts, _, _ = cv2.QRCodeDetector().detectAndDecodeMulti(img)
        return [t for t in (texts or []) if t] if ok else []
    except ImportError:
        raise RuntimeError("No QR decoder installed (pip install zxing-cpp)")


class MultipleCodes(Exception):
    """More than one different QR code was in view for the whole scan (e.g. a sheet of team codes)."""


def scan(get_frame: Callable[[], object], timeout_s: float = 6.0, interval_s: float = 0.15) -> str | None:
    """Grab camera frames until a QR code is read or the time runs out."""
    end = time.monotonic() + timeout_s
    saw_many = False
    while time.monotonic() < end:
        frame = get_frame()
        if frame is not None:
            found = list(dict.fromkeys(decode(frame)))
            if len(found) == 1:
                return found[0]
            saw_many = saw_many or len(found) > 1     # never guess between several codes
        time.sleep(interval_s)
    if saw_many:
        raise MultipleCodes()
    return None


_EVENT = re.compile(r"^([a-z0-9]{4,})$", re.I)       # 2026mokc, demo3528, ...


def parse_frc(text: str) -> dict:
    """Pull an FRC event key and team number out of a Nexus or The Blue Alliance link, if present."""
    out: dict = {"source": "", "event": "", "team": ""}
    try:
        u = urlparse(text.strip())
    except ValueError:
        return out
    host = (u.hostname or "").lower()
    if host.endswith("frc.nexus"):
        out["source"] = "nexus"
    elif host.endswith("thebluealliance.com"):
        out["source"] = "tba"
    else:
        return out
    parts = [p for p in u.path.split("/") if p]
    q = {k.lower(): v[0] for k, v in parse_qs(u.query).items() if v}
    for i, p in enumerate(parts):
        nxt = parts[i + 1] if i + 1 < len(parts) else ""
        if p.lower() == "event" and _EVENT.match(nxt):
            out["event"] = nxt.lower()
        elif p.lower() == "team" and nxt:
            m = re.match(r"(?:frc)?(\d{1,5})$", nxt, re.I)
            if m:
                out["team"] = m.group(1)
    # Real Nexus team codes: https://frc.nexus/<event>/<team>?key=<secret>   (e.g. /demo3528/1300?key=...)
    if out["source"] == "nexus" and not (out["event"] or out["team"]) and len(parts) == 2 \
            and _EVENT.match(parts[0]) and re.fullmatch(r"\d{1,5}", parts[1]):
        out["event"], out["team"] = parts[0].lower(), parts[1]
    out["event"] = out["event"] or (q.get("event", "").lower() if _EVENT.match(q.get("event", "")) else "")
    if not out["team"] and re.fullmatch(r"\d{1,5}", q.get("team", "")):
        out["team"] = q["team"]
    return out
