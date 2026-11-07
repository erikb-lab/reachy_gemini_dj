"""Speak phrases with Pocket TTS (raw digits vs reachy_brain.speakable), transcribe them back with Whisper, print both.
Runs ON THE LAPTOP (loads models):  python -s tools/tts_number_check.py"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_brain.speakable import speakable
from reachy_brain.stt import STT
from reachy_brain.tts import PocketTTS

PHRASES = ["Team 254 is ranked 3rd with a record of 9-1-0.", "Your next match is Qualification 12 at 3:30 pm.",
           "Team 1678 won in 2025.", "Volume is now 15%.", "Team 1005 is in pit 42.", "We scored 128 points, up 12.5%.",
           "The event has 3,600 teams."]
tts, stt = PocketTTS("alba", 4), STT("base.en", 4)


def say_and_hear(text):
    audio = np.concatenate([c for c in tts.stream(text)])
    a16 = np.interp(np.linspace(0, len(audio) - 1, int(len(audio) * 16000 / 24000)), np.arange(len(audio)), audio).astype(np.float32)
    return stt.transcribe(a16).strip()


for p in PHRASES:
    sp = speakable(p)
    print(f"TEXT      : {p}\nSPOKEN AS : {sp}\nHEARD(new): {say_and_hear(sp)}\nHEARD(raw): {say_and_hear(p)}\n", flush=True)
