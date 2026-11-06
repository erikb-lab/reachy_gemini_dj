"""Text to speech engines. Every engine yields float32 mono arrays at SAMPLE_RATE."""
from __future__ import annotations

from typing import Iterator

import numpy as np

SAMPLE_RATE = 24000


class TTS:
    sample_rate = SAMPLE_RATE

    def stream(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError


class PocketTTS(TTS):
    def __init__(self, voice: str = "alba", threads: int = 3):
        import torch
        torch.set_num_threads(threads)
        from pocket_tts import TTSModel
        self.model = TTSModel.load_model()
        self.state = self.model.get_state_for_audio_prompt(voice)
        list(self.model.generate_audio_stream(self.state, "Warm up."))

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for ch in self.model.generate_audio_stream(self.state, text):
            yield np.asarray(ch, dtype=np.float32).reshape(-1)


def make_tts(engine: str, voice: str, threads: int) -> TTS:
    if engine == "pocket":
        return PocketTTS(voice, threads)
    raise ValueError(f"unknown tts engine {engine!r} (available: pocket)")
