"""Text to speech engines. Every engine yields float32 mono arrays at SAMPLE_RATE."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import numpy as np

SAMPLE_RATE = 24000


class TTS:
    sample_rate = SAMPLE_RATE
    can_clone = False

    def stream(self, text: str) -> Iterator[np.ndarray]:
        raise NotImplementedError

    def set_voice(self, spec) -> None:
        """Switch voice: a Path to a saved voice file, or None for the engine's own voice."""
        raise NotImplementedError("this voice engine has no other voices")

    def clone(self, wav_path: Path, out_path: Path) -> None:
        """Make a voice file from a short recording of someone speaking."""
        raise NotImplementedError("this voice engine cannot clone a voice")


class PocketTTS(TTS):
    def __init__(self, voice: str = "alba", threads: int = 3):
        import torch
        torch.set_num_threads(threads)
        from pocket_tts import TTSModel
        self.model = TTSModel.load_model()
        self.default_voice = voice
        self._lock = threading.Lock()        # speaking and cloning share one model: one at a time
        self.state = self.model.get_state_for_audio_prompt(voice)
        list(self.model.generate_audio_stream(self.state, "Warm up."))

    @property
    def can_clone(self) -> bool:
        """True only when the model with voice cloning was downloaded (it needs the model terms accepted and a login)."""
        return bool(getattr(self.model, "has_voice_cloning", False))

    def stream(self, text: str) -> Iterator[np.ndarray]:
        with self._lock:
            for ch in self.model.generate_audio_stream(self.state, text):
                yield np.asarray(ch, dtype=np.float32).reshape(-1)

    def set_voice(self, spec) -> None:
        with self._lock:
            self.state = self.model.get_state_for_audio_prompt(Path(spec) if spec else self.default_voice)

    def clone(self, wav_path: Path, out_path: Path) -> None:
        if not self.can_clone:
            raise ValueError("Voice cloning is not set up on the brain. The Kyutai voice model needs its terms accepted "
                             "on Hugging Face and a read login on the brain computer; see the README.")
        from pocket_tts.models.model_state import export_model_state
        with self._lock:
            state = self.model.get_state_for_audio_prompt(Path(wav_path), truncate=True)
            export_model_state(state, out_path)


def make_tts(engine: str, voice: str, threads: int) -> TTS:
    if engine == "pocket":
        return PocketTTS(voice, threads)
    raise ValueError(f"unknown tts engine {engine!r} (available: pocket)")
