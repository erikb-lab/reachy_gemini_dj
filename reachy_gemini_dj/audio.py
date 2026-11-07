"""Audio pipeline for microphone capture, Gemini Live streaming, ducking, and echo protection."""
from __future__ import annotations

import collections
import logging
import math
import shutil
import struct
import subprocess
import threading
import time
from typing import Callable

import numpy as np

log = logging.getLogger(__name__)

GEMINI_IN_SR = 16000    # Gemini Live input sample rate
GEMINI_OUT_SR = 24000   # Gemini Live output sample rate
FRAME_S = 0.02          # 20ms playback frames


def to_float_mono(x) -> np.ndarray:
    """Convert raw SDK audio chunk / buffer to float32 mono array in [-1.0, 1.0]."""
    if isinstance(x, (bytes, bytearray)):
        count = len(x) // 2
        if count == 0:
            return np.zeros(0, dtype=np.float32)
        arr = np.frombuffer(x[:count * 2], dtype="<i2").astype(np.float32) / 32768.0
        return arr

    x = np.asarray(x)
    if x.dtype.kind in "iu":
        x = x.astype(np.float32) / 32768.0
    else:
        x = x.astype(np.float32, copy=False)

    if x.ndim > 1:
        # If shape is (samples, channels) or (channels, samples)
        if x.shape[0] <= 8 and x.shape[1] > x.shape[0]:
            x = x.mean(axis=0)
        else:
            x = x.mean(axis=1)
    return x


def resample(x: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    """Linear resampling for audio arrays."""
    if src_sr == dst_sr or len(x) < 2:
        return x
    target_len = max(1, int(round(len(x) * dst_sr / src_sr)))
    return np.interp(
        np.linspace(0, len(x) - 1, target_len),
        np.arange(len(x)),
        x,
    ).astype(np.float32)


# ---------------------------------------------------------------------------------------------------------
# Push-to-talk
# ---------------------------------------------------------------------------------------------------------
PTT_PREROLL_S = 0.30      # audio kept from just before the press so the first word is not clipped
PTT_TAIL_S = 0.90         # silence sent after release: closes the brain's VAD (650 ms) and Gemini's activity detection
PTT_MAX_HOLD_S = 30.0     # safety: a stuck button / lost "up" event releases itself
LISTEN_MODES = ("always", "push_to_talk")


class PttGate:
    """Decides which mic frames reach the voice backend in push-to-talk mode. Pure logic, no hardware, thread-safe.

    Idle:  frames are dropped but the last ~300 ms are remembered (pre-roll).
    Held:  frames are passed through; the pre-roll is delivered ahead of the first one after the press.
    Tail:  after release, ~0.9 s of silence (same length as each incoming frame, so it is paced in real time)
           is sent instead of the mic, then the gate goes idle again.
    All frames are 16 kHz 16-bit mono PCM bytes. `clock` is injectable for tests.
    """

    def __init__(self, preroll_s: float = PTT_PREROLL_S, tail_s: float = PTT_TAIL_S,
                 max_hold_s: float = PTT_MAX_HOLD_S, sample_rate: int = GEMINI_IN_SR, clock=time.monotonic):
        self._preroll_bytes = int(preroll_s * sample_rate) * 2
        self._tail_bytes = int(tail_s * sample_rate) * 2
        self._max_hold_s = max_hold_s
        self._clock = clock
        self._lock = threading.Lock()
        self._preroll: collections.deque[bytes] = collections.deque()
        self._preroll_len = 0
        self._held = False
        self._since = 0.0
        self._deliver_preroll = False
        self._tail_left = 0

    def _expire(self) -> None:          # caller holds the lock
        if self._held and self._clock() - self._since > self._max_hold_s:
            log.warning("push-to-talk: auto-released after %.0f s held", self._max_hold_s)
            self._release()

    def _release(self) -> None:
        if not self._held:
            return
        self._held = False
        self._deliver_preroll = False
        self._tail_left = self._tail_bytes
        self._preroll.clear()
        self._preroll_len = 0

    @property
    def active(self) -> bool:
        with self._lock:
            self._expire()
            return self._held

    def press(self) -> bool:
        """Start talking. Returns True if this was a new press (not already held)."""
        with self._lock:
            self._expire()
            if self._held:
                self._since = self._clock()
                return False
            self._held = True
            self._since = self._clock()
            self._tail_left = 0
            self._deliver_preroll = True
            return True

    def release(self) -> bool:
        """Stop talking. Returns True if it was held."""
        with self._lock:
            was = self._held
            self._release()
            return was

    def toggle(self) -> bool:
        """Tap-to-start / tap-to-stop (one-button devices). Returns the new held state."""
        if self.active:
            self.release()
            return False
        self.press()
        return True

    def reset(self) -> None:
        with self._lock:
            self._held = False
            self._deliver_preroll = False
            self._tail_left = 0
            self._preroll.clear()
            self._preroll_len = 0

    def process(self, pcm: bytes) -> list[bytes]:
        """Feed one mic frame; returns the frames (possibly none) to send to the backend, in order."""
        with self._lock:
            self._expire()
            if self._held:
                out = []
                if self._deliver_preroll:
                    out.extend(self._preroll)
                    self._preroll.clear()
                    self._preroll_len = 0
                    self._deliver_preroll = False
                out.append(pcm)
                return out
            if self._tail_left > 0:
                self._tail_left -= len(pcm)
                return [bytes(len(pcm))]
            self._preroll.append(pcm)
            self._preroll_len += len(pcm)
            while len(self._preroll) > 1 and self._preroll_len - len(self._preroll[0]) >= self._preroll_bytes:
                self._preroll_len -= len(self._preroll.popleft())
            return []


def set_hw_speaker_level(percent: int) -> str:
    """Set the Reachy Mini speaker's ALSA 'PCM' playback level (stereo front control). Returns a status string.
    The robot ships at 72% (-17 dB), which makes music quiet no matter what the app's gain says."""
    if not percent:
        return "left alone"
    amixer = shutil.which("amixer")
    if not amixer:
        return "amixer not found"
    card = "0"
    try:
        for line in open("/proc/asound/cards"):
            if "Reachy Mini Audio" in line:
                card = line.split()[0]
                break
    except OSError:
        pass
    pct = max(0, min(100, int(percent)))
    try:
        r = subprocess.run([amixer, "-c", card, "sset", "PCM,0", f"{pct}%"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError) as e:
        return f"amixer failed: {e}"
    return f"speaker level {pct}%" if r.returncode == 0 else f"amixer error: {r.stderr.strip()[:80]}"


class AudioPipeline:
    """Manages microphone input, adaptive noise thresholding, speech ducking, and voice playback."""

    def __init__(
        self,
        mini,
        stop: threading.Event,
        config,
        music,
        on_level: Callable[[float], None] | None = None,
        on_pcm_chunk: Callable[[bytes], None] | None = None,
        on_talk_pressed: Callable[[], None] | None = None,
    ):
        self.mini = mini
        self.stop = stop
        self.config = config
        self.music = music
        self.on_level = on_level or (lambda lvl: None)
        self.on_pcm_chunk = on_pcm_chunk
        # Called on every talk-button press (also in "always" mode); e.g. to dismiss an alarm. May be set later.
        self.on_talk_pressed = on_talk_pressed
        self._ptt = PttGate()
        self._ptt_was_held = False

        # Hardware media interface
        self.media = getattr(mini, "media", mini)

        # Sampling rates & channels
        self.in_sr = self._query_media_int("get_input_audio_samplerate", 16000)
        self.out_sr = self._query_media_int("get_output_audio_samplerate", 16000)
        self.out_ch = max(1, self._query_media_int("get_output_channels", 2))

        # Audio state
        self._muted = False
        self._user_speaking = False
        self._last_user_speech = -1e9        # "never": monotonic() is time since boot, so 0.0 would look recent

        # Adaptive noise floor & echo suppression
        self.noise_floor = 0.012
        self.smoothed_level = 0.0
        self.speaker_level = 0.0

        # Voice playback queue (Gemini Live speech chunks)
        self._voice_queue: collections.deque[np.ndarray] = collections.deque()
        self._voice_lock = threading.Lock()
        self._is_robot_speaking = False
        self._out_shape = "2d"

        # Worker threads
        self._capture_thread: threading.Thread | None = None
        self._playback_thread: threading.Thread | None = None

    def _query_media_int(self, method_name: str, default: int) -> int:
        try:
            fn = getattr(self.media, method_name, None)
            val = fn() if callable(fn) else fn
            return int(val) if val else default
        except Exception:
            return default

    # ---------- push-to-talk (see PttGate); the API a physical button calls through main.py's /api/ptt ----------
    @property
    def listen_mode(self) -> str:
        m = str(getattr(self.config, "listen_mode", "always") or "always").lower()
        return m if m in LISTEN_MODES else "always"

    @property
    def ptt_active(self) -> bool:
        return self.listen_mode == "push_to_talk" and self._ptt.active

    def _talk_pressed(self) -> None:
        """Pressing talk while the robot speaks is a barge-in; then tell the optional hook (alarm dismiss etc.)."""
        if self._is_robot_speaking:
            self.clear_voice()
        cb = self.on_talk_pressed
        if cb:
            try:
                cb()
            except Exception as e:
                log.warning("on_talk_pressed hook failed: %s", e)

    def ptt_set(self, down: bool) -> bool:
        """Talk button down/up. In "always" mode only a press has an effect (barge-in + hook); audio is untouched.
        Returns whether talk is now held."""
        if down:
            new = self._ptt.press() if self.listen_mode == "push_to_talk" else True
            if new:
                self._talk_pressed()
                if self.listen_mode == "push_to_talk":
                    self.music.set_duck(self.config.duck_level)
        else:
            self._ptt.release()
            if not self._user_speaking and not self._is_robot_speaking:
                self.music.reset_duck()
        return self.ptt_active

    def ptt_toggle(self) -> bool:
        return self.ptt_set(not self.ptt_active)

    def set_mute(self, muted: bool) -> None:
        self._muted = bool(muted)

    def play_voice(self, pcm_24k: bytes) -> None:
        """Receive 24 kHz 16-bit mono PCM bytes from Gemini Live and queue for playback."""
        if not pcm_24k:
            return
        arr = np.frombuffer(pcm_24k[: len(pcm_24k) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        if self.out_sr != GEMINI_OUT_SR:
            arr = resample(arr, GEMINI_OUT_SR, self.out_sr)
        with self._voice_lock:
            self._voice_queue.append(arr)
            self._is_robot_speaking = True
        # Duck music while robot is speaking
        self.music.set_duck(self.config.duck_level)

    def clear_voice(self) -> None:
        """Clear playback queue immediately on barge-in / interruption."""
        with self._voice_lock:
            self._voice_queue.clear()
            self._is_robot_speaking = False
        if not self._user_speaking:
            self.music.reset_duck()

    def start(self) -> None:
        log.info("Hardware speaker: %s", set_hw_speaker_level(getattr(self.config, "hw_speaker_percent", 100)))
        # Start hardware streams via media helper
        try:
            if hasattr(self.media, "start_recording"):
                self.media.start_recording()
        except Exception as e:
            log.warning("Could not start hardware recording: %s", e)

        try:
            if hasattr(self.media, "start_playing"):
                self.media.start_playing()
        except Exception as e:
            log.warning("Could not start hardware playback: %s", e)

        self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True, name="audio-capture")
        self._playback_thread = threading.Thread(target=self._playback_loop, daemon=True, name="audio-playback")
        self._capture_thread.start()
        self._playback_thread.start()
        log.info("AudioPipeline started (in=%d Hz, out=%d Hz x %d ch)", self.in_sr, self.out_sr, self.out_ch)

    def stop(self) -> None:
        self.clear_voice()
        try:
            if hasattr(self.media, "stop_recording"):
                self.media.stop_recording()
        except Exception:
            pass
        try:
            if hasattr(self.media, "stop_playing"):
                self.media.stop_playing()
        except Exception:
            pass

    def _capture_loop(self) -> None:
        while not self.stop.is_set():
            if self._muted:
                self.on_level(0.0)
                time.sleep(0.05)
                continue

            try:
                sample = None
                if hasattr(self.media, "get_audio_sample"):
                    sample = self.media.get_audio_sample()

                if sample is None or len(sample) == 0:
                    time.sleep(0.005)
                    continue

                mono = to_float_mono(sample)
                if len(mono) == 0:
                    continue

                # Resample to 16 kHz for Gemini input if needed
                x16 = resample(mono, self.in_sr, GEMINI_IN_SR) if self.in_sr != GEMINI_IN_SR else mono

                # Calculate RMS
                rms = float(np.sqrt(np.mean(np.square(x16))))
                self.smoothed_level = 0.8 * self.smoothed_level + 0.2 * rms
                self.on_level(self.smoothed_level)

                # Adaptive noise floor estimation
                now = time.monotonic()
                if not self._user_speaking and not self._is_robot_speaking:
                    # Slow exponential tracking of ambient background noise
                    self.noise_floor = 0.99 * self.noise_floor + 0.01 * rms

                # Adaptive threshold for voice activity detection
                # Base speech threshold requires clear margin above background noise
                echo_factor = max(0.5, float(self.config.echo_guard))
                base_speech_thr = max(0.015, self.noise_floor * 2.2 * echo_factor)

                # If robot speaker is playing, raise threshold to suppress echo
                # Music from the robot's speaker leaks into the mic just like its voice does: without this the
                # music itself counts as "speech" (music stays ducked, lyrics get transcribed, barge-ins fire).
                music_on = self._music_audible()
                if self._is_robot_speaking:
                    threshold = max(base_speech_thr, 0.06 * echo_factor)
                else:
                    threshold = base_speech_thr
                if music_on:
                    threshold = max(threshold, 0.06 * echo_factor * max(0.5, float(self.music._volume)))

                is_speech = rms > threshold

                if is_speech:
                    self._user_speaking = True
                    self._last_user_speech = now + 0.7  # Hangover time
                    self.music.set_duck(self.config.duck_level)
                elif self._user_speaking and now > self._last_user_speech:
                    self._user_speaking = False
                    if not self._is_robot_speaking:
                        self.music.reset_duck()

                # Push-to-talk: only frames the gate lets through are sent ("always" mode skips this block)
                ptt_on = self.listen_mode == "push_to_talk"
                if not ptt_on and self._ptt.active:
                    self._ptt.reset()                  # switched back to "always" while held
                if self.on_pcm_chunk and ptt_on:
                    # The user is deliberately talking, so no echo attenuation (music is ducked on press instead)
                    raw = (np.clip(x16, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    for chunk in self._ptt.process(raw):
                        self.on_pcm_chunk(chunk)
                    held = self._ptt.active
                    if self._ptt_was_held and not held and not self._user_speaking and not self._is_robot_speaking:
                        self.music.reset_duck()        # released by the 30 s safety, not by the button
                    self._ptt_was_held = held
                # Stream audio to Gemini Live if callback registered
                elif self.on_pcm_chunk:
                    # Echo suppression: if robot is speaking and user is NOT clearly interrupting,
                    # attenuate mic input so Gemini doesn't hear its own echo
                    if (self._is_robot_speaking or music_on) and not (rms > 0.12 * echo_factor):
                        out_pcm = (np.clip(x16 * 0.05, -1.0, 1.0) * 32767).astype("<i2").tobytes()
                    else:
                        out_pcm = (np.clip(x16, -1.0, 1.0) * 32767).astype("<i2").tobytes()

                    self.on_pcm_chunk(out_pcm)

            except Exception as e:
                log.debug("Audio capture loop iteration note: %s", e)
                time.sleep(0.02)

    def _music_audible(self) -> bool:
        try:
            return bool(self.music.is_playing)
        except Exception:
            return False

    def _playback_loop(self) -> None:
        chunk_size = int(self.out_sr * FRAME_S)
        while not self.stop.is_set():
            frame: np.ndarray | None = None
            with self._voice_lock:
                if self._voice_queue:
                    a = self._voice_queue[0]
                    if len(a) <= chunk_size:
                        frame = a
                        self._voice_queue.popleft()
                    else:
                        frame = a[:chunk_size]
                        self._voice_queue[0] = a[chunk_size:]
                else:
                    if self._is_robot_speaking:
                        self._is_robot_speaking = False
                        if not self._user_speaking:
                            self.music.reset_duck()

            if frame is None:
                time.sleep(0.01)
                continue

            # Scale by voice gain
            gain = max(0.1, min(3.0, float(self.config.voice_gain)))
            frame = np.clip(frame * gain, -1.0, 1.0)

            # Format for output channels
            if self.out_ch == 1:
                out_data = frame.astype(np.float32)
            else:
                out_data = np.column_stack([frame] * self.out_ch).astype(np.float32)

            try:
                if hasattr(self.media, "push_audio_sample"):
                    payload = out_data if self._out_shape == "2d" else frame
                    self.media.push_audio_sample(payload)
            except Exception as e:
                if self._out_shape == "2d":
                    self._out_shape = "1d"
                    try:
                        self.media.push_audio_sample(frame)
                    except Exception:
                        pass
                else:
                    log.debug("push_audio_sample note: %s", e)
