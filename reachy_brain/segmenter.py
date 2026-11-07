"""Cut a continuous 16 kHz mono int16 mic stream into utterances using WebRTC VAD."""
from __future__ import annotations

import collections

import numpy as np

SR = 16000
FRAME = 480            # 30 ms
FRAME_BYTES = FRAME * 2


class Segmenter:
    """feed(bytes) returns a list of events: ("start", None) when speech begins and ("utterance", float32 array)
    when it ends. Keeps ~300 ms of pre-roll so the first syllable is not clipped."""

    def __init__(self, aggressiveness: int = 2, end_silence_ms: int = 650, max_utt_s: float = 20.0,
                 min_speech_ms: int = 240, vad=None):
        if vad is None:
            import webrtcvad
            vad = webrtcvad.Vad(aggressiveness)
        self.vad = vad
        self.end_frames = max(1, end_silence_ms // 30)
        self.max_frames = int(max_utt_s * 1000 / 30)
        self.min_speech_frames = max(1, min_speech_ms // 30)
        self._buf = b""
        self._pre: collections.deque[bytes] = collections.deque(maxlen=10)
        self._recent: collections.deque[bool] = collections.deque(maxlen=5)
        self._frames: list[bytes] = []
        self._speaking = False
        self._silence = 0
        self._voiced = 0

    @property
    def speaking(self) -> bool:
        return self._speaking

    @property
    def voiced_frames(self) -> int:
        """Voiced 30 ms frames in the utterance so far (0 when idle)."""
        return self._voiced if self._speaking else 0

    def reset(self) -> None:
        self._buf = b""
        self._pre.clear()
        self._recent.clear()
        self._frames = []
        self._speaking = False
        self._silence = self._voiced = 0

    def feed(self, pcm: bytes) -> list[tuple[str, object]]:
        out: list[tuple[str, object]] = []
        self._buf += pcm
        while len(self._buf) >= FRAME_BYTES:
            frame, self._buf = self._buf[:FRAME_BYTES], self._buf[FRAME_BYTES:]
            voiced = bool(self.vad.is_speech(frame, SR))
            if not self._speaking:
                self._pre.append(frame)
                self._recent.append(voiced)
                if sum(self._recent) >= 3:
                    self._speaking = True
                    self._frames = list(self._pre)
                    self._pre.clear()
                    self._silence = 0
                    self._voiced = sum(self._recent)
                    out.append(("start", None))
            else:
                self._frames.append(frame)
                if voiced:
                    self._silence = 0
                    self._voiced += 1
                else:
                    self._silence += 1
                if self._silence >= self.end_frames or len(self._frames) >= self.max_frames:
                    frames, voiced_n = self._frames, self._voiced
                    self.reset()
                    if voiced_n >= self.min_speech_frames:
                        a = np.frombuffer(b"".join(frames), dtype="<i2").astype(np.float32) / 32768.0
                        out.append(("utterance", a))
                    else:
                        out.append(("cancel", None))
        return out
