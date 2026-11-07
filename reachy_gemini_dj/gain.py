"""Live gain control for the streamed music pipeline.

Volume scale (single source of truth, used by music.py, gemini.py and main.py)
------------------------------------------------------------------------------
``music_volume`` is a *linear amplitude multiplier*:

    0.0  = silent
    1.0  = unity gain (100 %, the track's own level)
    2.5  = software boost (250 %), the maximum (``MAX_VOLUME``)

The UI shows it as a percentage (value * 100).  The Gemini ``set_volume`` tool
takes the same percentage (0-250).  ``duck_level`` is a separate multiplier in
0.0-1.0 applied on top of the volume while someone is talking.

Why this module exists
----------------------
ffplay fixes its volume at launch, so gain cannot change mid-track.  Instead
ffmpeg decodes the stream to raw s16le PCM on a pipe, ``GainPump`` applies the
(smoothly ramped) gain with numpy on small blocks, and writes to ffplay reading
raw PCM from stdin.  Nothing is written to disk.
"""
from __future__ import annotations

import functools
import logging
import functools
import re
import subprocess
from typing import BinaryIO, Callable

import numpy as np

log = logging.getLogger(__name__)

MIN_VOLUME = 0.0
MAX_VOLUME = 2.5
DEFAULT_VOLUME = 1.0

SAMPLE_RATE = 44100
CHANNELS = 2
BLOCK_MS = 20  # gain is re-evaluated and ramped once per block

# Duck envelope: how long (seconds) a full 0 -> 1 gain swing takes.
ATTACK_S = 0.08   # gain falling (duck in): fast so speech is clear
RELEASE_S = 0.45  # gain rising (duck out / volume up): slow so no pumping


def clamp_volume(v: float) -> float:
    """Clamp a music volume multiplier to [MIN_VOLUME, MAX_VOLUME]; NaN -> default."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return DEFAULT_VOLUME
    if v != v:  # NaN
        return DEFAULT_VOLUME
    return max(MIN_VOLUME, min(MAX_VOLUME, v))


def clamp_duck(v: float) -> float:
    """Clamp a duck multiplier to [0, 1]; NaN -> 1 (no ducking)."""
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 1.0
    if v != v:
        return 1.0
    return max(0.0, min(1.0, v))


def percent_to_volume(pct: float) -> float:
    """Convert a 0-250 percent value (tool / UI) to a clamped multiplier."""
    try:
        return clamp_volume(float(pct) / 100.0)
    except (TypeError, ValueError):
        return DEFAULT_VOLUME


def volume_to_percent(vol: float) -> int:
    return int(round(clamp_volume(vol) * 100))


def slew(current: float, target: float, dt: float) -> float:
    """Move ``current`` toward ``target`` limited by attack/release rates.

    Falling uses ATTACK_S, rising uses RELEASE_S (per unit of gain per second).
    """
    if current == target:
        return current
    if target < current:
        return max(target, current - dt / ATTACK_S)
    return min(target, current + dt / RELEASE_S)


def apply_gain(pcm: bytes, g0: float, g1: float, channels: int = CHANNELS) -> bytes:
    """Scale interleaved s16le PCM by a gain ramping linearly g0 -> g1.

    Vectorised numpy; clips to int16 range (boost above unity can clip).
    Trailing partial frames are dropped.
    """
    frame_bytes = 2 * channels
    pcm = pcm[: len(pcm) // frame_bytes * frame_bytes]
    if not pcm:
        return b""
    if g0 == 1.0 and g1 == 1.0:
        return pcm
    if g0 == 0.0 and g1 == 0.0:
        return bytes(len(pcm))
    x = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels).astype(np.float32)
    n = x.shape[0]
    if g0 == g1:
        x *= g1
    else:
        ramp = np.linspace(g0, g1, n + 1, dtype=np.float32)[1:]
        x *= ramp[:, None]
    np.clip(x, -32768.0, 32767.0, out=x)
    return x.astype("<i2").tobytes()


class GainPump:
    """Reads PCM from ``src``, applies the live gain, writes it to ``dst``.

    ``get_target`` returns the desired total gain (volume * duck) and is called
    once per block, so changes take effect within one block plus whatever audio
    is already buffered in the pipes / ffplay (a few hundred ms).
    """

    def __init__(
        self,
        src: BinaryIO,
        dst: BinaryIO,
        get_target: Callable[[], float],
        *,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        block_ms: int = BLOCK_MS,
        initial_gain: float | None = None,
    ):
        self.src = src
        self.dst = dst
        self.get_target = get_target
        self.channels = channels
        self.sample_rate = sample_rate
        self.block_frames = max(1, sample_rate * block_ms // 1000)
        self.block_bytes = self.block_frames * 2 * channels
        self.gain = float(get_target() if initial_gain is None else initial_gain)

    def process_block(self, pcm: bytes) -> bytes:
        target = self.get_target()
        g0 = self.gain
        g1 = slew(g0, target, len(pcm) / (2 * self.channels) / self.sample_rate)
        self.gain = g1
        return apply_gain(pcm, g0, g1, self.channels)

    def run(self) -> None:
        try:
            while True:
                buf = self.src.read(self.block_bytes)
                if not buf:
                    break
                out = self.process_block(buf)
                if out:
                    self.dst.write(out)
                    self.dst.flush()
        except (BrokenPipeError, ValueError, OSError):
            pass  # sink or decoder went away (skip/stop); caller cleans up
        finally:
            try:
                self.dst.close()
            except Exception:
                pass


@functools.lru_cache(maxsize=1)
@functools.lru_cache(maxsize=1)
def _ffplay_major() -> int:
    try:
        out = subprocess.run(["ffplay", "-version"], capture_output=True, text=True, timeout=5).stdout
        m = re.search(r"version\D*(\d+)", out)
        return int(m.group(1)) if m else 99
    except Exception:
        return 99


def sink_command() -> list[str]:
    """ffplay reading raw s16le PCM from stdin (same audio output path as before)."""
    layout = ["-ch_layout", "stereo"] if _ffplay_major() >= 7 else ["-channels", str(CHANNELS)]
    return [
        "ffplay", "-nodisp", "-autoexit", "-nostats", "-loglevel", "warning",
        "-fflags", "nobuffer", "-probesize", "32", "-analyzeduration", "0",
        "-f", "s16le", "-sample_rate", str(SAMPLE_RATE), *layout, "-i", "pipe:0",
    ]


def decoder_command(stream_url: str, headers: dict[str, str] | None = None) -> list[str]:
    """ffmpeg decoding the remote stream to raw s16le PCM on stdout."""
    cmd = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "warning", "-vn"]
    if headers:
        cmd += ["-headers", "".join(f"{k}: {v}\r\n" for k, v in headers.items())]
    cmd += [
        "-reconnect", "1", "-reconnect_streamed", "1", "-reconnect_delay_max", "5",
        "-i", stream_url,
        "-f", "s16le", "-ac", str(CHANNELS), "-ar", str(SAMPLE_RATE), "pipe:1",
    ]
    return cmd


def shrink_pipe(fileobj, size: int = 8192) -> None:
    """Best effort: shrink a pipe buffer to cut gain-change latency (Linux only)."""
    try:
        import fcntl

        fcntl.fcntl(fileobj.fileno(), 1031, size)  # F_SETPIPE_SZ
    except Exception:
        pass
