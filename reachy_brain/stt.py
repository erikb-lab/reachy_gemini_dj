"""Speech to text with faster-whisper."""
from __future__ import annotations

import numpy as np


class STT:
    def __init__(self, model: str = "base.en", threads: int = 3):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        self.english_only = model.endswith(".en")
        self.transcribe(np.zeros(8000, dtype=np.float32))     # warm up

    def transcribe(self, audio_16k: np.ndarray) -> str:
        kw = {"language": "en"} if self.english_only else {}
        segs, _ = self.model.transcribe(audio_16k, beam_size=1, vad_filter=False, **kw)
        return " ".join(s.text.strip() for s in segs).strip()
