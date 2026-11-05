"""Audio pipeline for microphone capture, ducking, and echo protection."""
from __future__ import annotations

import logging
import math
import struct
import threading
import time
from typing import Callable

log = logging.getLogger(__name__)


class AudioPipeline:
    def __init__(
        self,
        mini,
        stop: threading.Event,
        config,
        music,
        on_level: Callable[[float], None],
    ):
        self.mini = mini
        self.stop = stop
        self.config = config
        self.music = music
        self.on_level = on_level
        self._muted = False
        self._thread: threading.Thread | None = None
        self._is_speaking = False

    def set_mute(self, muted: bool) -> None:
        self._muted = bool(muted)

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="audio-pipeline")
        self._thread.start()

    def stop_pipeline(self) -> None:
        pass

    def _run(self) -> None:
        try:
            if hasattr(self.mini, "start_recording"):
                self.mini.start_recording()
        except Exception as e:
            log.warning("Could not start recording on robot: %s", e)

        while not self.stop.is_set():
            if self._muted:
                self.on_level(0.0)
                time.sleep(0.1)
                continue

            try:
                sample = None
                if hasattr(self.mini, "get_audio_sample"):
                    sample = self.mini.get_audio_sample()
                if sample is not None and len(sample) > 0:
                    # Calculate RMS
                    rms = self._calc_rms(sample)
                    self.on_level(rms)

                    # Dynamic audio ducking
                    threshold = 0.03 * self.config.echo_guard
                    if rms > threshold:
                        if not self._is_speaking:
                            self._is_speaking = True
                            self.music.set_duck(self.config.duck_level)
                    else:
                        if self._is_speaking:
                            self._is_speaking = False
                            self.music.reset_duck()
                else:
                    time.sleep(0.05)
            except Exception as e:
                log.debug("Sample read note: %s", e)
                time.sleep(0.1)

    @staticmethod
    def _calc_rms(data: bytes | list | tuple) -> float:
        if isinstance(data, (bytes, bytearray)):
            count = len(data) // 2
            if count == 0:
                return 0.0
            shorts = struct.unpack(f"{count}h", data[: count * 2])
            sum_sq = sum(s * s for s in shorts)
            return math.sqrt(sum_sq / count) / 32768.0
        return 0.0
