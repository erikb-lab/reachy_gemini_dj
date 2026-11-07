"""Speech to text with faster-whisper."""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)


class STT:
    def __init__(self, model: str = "base.en", threads: int = 3):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=threads)
        self.english_only = model.endswith(".en")
        self.transcribe(np.zeros(8000, dtype=np.float32))     # warm up

    def transcribe(self, audio_16k: np.ndarray) -> str:
        kw = {"language": "en"} if self.english_only else {}
        # condition_on_previous_text=False stops Whisper's "yes, cool. yes, cool..." loops; low-confidence segments
        # (noise, music, the robot's own voice) are dropped rather than turned into random words.
        segs, _ = self.model.transcribe(audio_16k, beam_size=1, vad_filter=False, condition_on_previous_text=False, **kw)
        kept = []
        for s in segs:
            t = s.text.strip()
            if not t:
                continue
            if (s.no_speech_prob > 0.6 and s.avg_logprob < -0.7) or s.avg_logprob < -1.1 or s.compression_ratio > 2.4:
                log.info("stt: dropped low-confidence segment %r (no_speech=%.2f logprob=%.2f ratio=%.1f)",
                         t, s.no_speech_prob, s.avg_logprob, s.compression_ratio)
                continue
            kept.append(t)
        return " ".join(kept).strip()
