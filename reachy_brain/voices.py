"""The brain's list of voices: make a new voice from a recording of someone who agreed, keep it, switch to it.

Only the small voice file the TTS engine makes from the recording is kept (not the recording itself), with a record of
who the voice belongs to and that the person who made it confirmed the speaker agreed. Everything lives under one
folder on the brain computer (default ~/.cache/reachy_brain/voices) and never leaves it.
"""
from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import Callable

log = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 5 * 1024 * 1024       # the robot -> brain WebSocket carries at most 8 MB per message, base64 included
MAX_VOICES = 20
MIN_SECONDS = 2.0                        # shorter samples do not carry enough of the voice
MAX_SECONDS = 30                         # the engine only uses the first 30 s anyway
NAME_MAX = 40
ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")     # no dots or slashes: an id can never leave the voices folder
BUILTIN_ID = "default"                   # the engine's own voice (config tts_voice, "alba")
WAV_BYTES_PER_SECOND = 24000 * 2         # mono 16-bit 24 kHz, what convert() writes


class VoiceError(Exception):
    """A problem the person using the control page can fix; the message is shown to them as is."""


def ffmpeg_to_wav(src: Path, dst: Path) -> None:
    """Any common audio file -> mono 24 kHz 16-bit wav, first MAX_SECONDS only (ffmpeg; the arguments are a list, so a
    file name can never become a command)."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src), "-vn", "-t", str(MAX_SECONDS),
           "-ac", "1", "-ar", "24000", "-sample_fmt", "s16", "-f", "wav", str(dst)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=60)
    except FileNotFoundError:
        raise VoiceError("The brain computer needs ffmpeg installed to read audio files.")
    except subprocess.TimeoutExpired:
        raise VoiceError("That audio file took too long to read. Try a shorter or simpler file.")
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
        raise VoiceError("I could not read that file as audio. Try an mp3, wav or m4a recording.")


def clean_name(name: object) -> str:
    s = re.sub(r"[\x00-\x1f\x7f]", " ", str(name or ""))
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        raise VoiceError("Give the voice a name (1 to 40 characters).")
    return s[:NAME_MAX]


class VoiceStore:
    def __init__(self, root: str | Path, convert: Callable[[Path, Path], None] = ffmpeg_to_wav):
        self.root = Path(os.path.expanduser(str(root)))
        self.convert = convert
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)

    # ------------------------------------------------------------------ reading
    def _meta_path(self, vid: str) -> Path:
        return self.root / f"{vid}.json"

    def path(self, vid: str) -> Path | None:
        if not isinstance(vid, str) or not ID_RE.match(vid):
            return None
        p = self.root / f"{vid}.safetensors"
        return p if p.is_file() else None

    def _read_meta(self, p: Path) -> dict:
        vid = p.stem
        try:
            m = json.loads(self._meta_path(vid).read_text())
            if isinstance(m, dict):
                m["id"] = vid
                return m
        except (OSError, ValueError):
            pass
        # A voice file that arrived without details (copied in by hand): list it, and say there is no consent record.
        return {"id": vid, "name": vid.replace("_", " ").replace("-", " ").title(), "created": None,
                "consent": {"confirmed": False, "note": "added without a consent record"}, "seconds": None}

    def list(self) -> list[dict]:
        out = []
        for p in sorted(self.root.glob("*.safetensors")):
            if ID_RE.match(p.stem):
                m = self._read_meta(p)
                out.append({"id": m["id"], "name": m.get("name") or m["id"], "created": m.get("created"),
                            "seconds": m.get("seconds"), "consent": m.get("consent") or {"confirmed": False}})
        out.sort(key=lambda v: (v["created"] or "", v["name"]))
        return out

    # ------------------------------------------------------------------ the voice in use
    def active(self) -> str:
        try:
            vid = json.loads((self.root / "active.json").read_text()).get("id")
        except (OSError, ValueError, AttributeError):
            return BUILTIN_ID
        return vid if vid == BUILTIN_ID or self.path(vid) else BUILTIN_ID

    def set_active(self, vid: str) -> None:
        if vid != BUILTIN_ID and not self.path(vid):
            raise VoiceError("That voice is not on the brain any more.")
        self._write_json(self.root / "active.json", {"id": vid})

    def spec(self, vid: str | None = None):
        """What the TTS engine should load: the voice file's Path, or None for the engine's own voice."""
        vid = vid or self.active()
        return None if vid == BUILTIN_ID else self.path(vid)

    # ------------------------------------------------------------------ changing
    def _write_json(self, p: Path, obj: dict) -> None:
        tmp = p.with_suffix(p.suffix + ".part")
        tmp.write_text(json.dumps(obj, indent=1))
        os.chmod(tmp, 0o600)
        tmp.replace(p)

    def create(self, name: object, consent: object, audio: bytes, filename: object,
               clone: Callable[[Path, Path], None]) -> dict:
        """Make a voice from a recording. `clone(wav_path, out_path)` is the TTS engine's cloning step."""
        if consent is not True:
            raise VoiceError("Tick the box to confirm the speaker agreed to their voice being used here.")
        name = clean_name(name)
        if not audio or len(audio) < 1000:
            raise VoiceError("That recording is empty or far too short.")
        if len(audio) > MAX_UPLOAD_BYTES:
            raise VoiceError("That file is too big (the limit is %d MB). A 30 second clip is plenty." % (MAX_UPLOAD_BYTES >> 20))
        if len(self.list()) >= MAX_VOICES:
            raise VoiceError("There are already %d voices. Delete one first." % MAX_VOICES)
        ext = re.sub(r"[^a-z0-9]", "", Path(str(filename or "")).suffix.lower())[:5] or "bin"
        work = Path(tempfile.mkdtemp(prefix=".work-", dir=self.root))
        try:
            src, wav, out = work / f"upload.{ext}", work / "sample.wav", work / "voice.safetensors"
            src.write_bytes(audio)
            self.convert(src, wav)
            seconds = round(wav.stat().st_size / WAV_BYTES_PER_SECOND, 1)
            if seconds < MIN_SECONDS:
                raise VoiceError("That recording is only %.1f seconds of sound. Use at least %d seconds of clear speech." % (seconds, MIN_SECONDS))
            src.unlink()                                        # the recording itself is not kept
            clone(wav, out)
            if not out.exists() or out.stat().st_size < 1000:
                raise VoiceError("The voice engine did not make a voice from that recording.")
            vid = uuid.uuid4().hex[:12]
            os.chmod(out, 0o600)
            out.replace(self.root / f"{vid}.safetensors")
            now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
            meta = {"id": vid, "name": name, "created": now, "seconds": min(seconds, MAX_SECONDS),
                    "consent": {"confirmed": True, "at": now, "by": "the person using the control page",
                                "statement": "The speaker agreed to their voice being used here."}}
            self._write_json(self._meta_path(vid), meta)
            log.info("voice created: %s (%s, %.1f s of sample)", vid, name, seconds)
            return {"id": vid, "name": name, "created": now, "seconds": meta["seconds"], "consent": meta["consent"]}
        finally:
            shutil.rmtree(work, ignore_errors=True)

    def delete(self, vid: str) -> bool:
        p = self.path(vid)
        if not p:
            return False
        was_active = self.active() == vid
        p.unlink()
        self._meta_path(vid).unlink(missing_ok=True)
        if was_active:                                          # back to the engine's own voice
            (self.root / "active.json").unlink(missing_ok=True)
        return True


def decode_audio(b64: object) -> bytes:
    try:
        return base64.b64decode(str(b64 or ""), validate=True)
    except (binascii.Error, ValueError):
        raise VoiceError("The upload was damaged on the way. Try again.")
