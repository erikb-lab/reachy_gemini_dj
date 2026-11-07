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
    ):
        self.mini = mini
        self.stop = stop
        self.config = config
        self.music = music
        self.on_level = on_level or (lambda lvl: None)
        self.on_pcm_chunk = on_pcm_chunk

        # Hardware media interface
        self.media = getattr(mini, "media", mini)

        # Sampling rates & channels
        self.in_sr = self._query_media_int("get_input_audio_samplerate", 16000)
        self.out_sr = self._query_media_int("get_output_audio_samplerate", 16000)
        self.out_ch = max(1, self._query_media_int("get_output_channels", 2))

        # Audio state
        self._muted = False
        self._user_speaking = False
        self._last_user_speech = 0.0

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

                # Stream audio to Gemini Live if callback registered
                if self.on_pcm_chunk:
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
