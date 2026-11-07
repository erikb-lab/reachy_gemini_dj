"""Round trip: speak phrases with Pocket TTS (after reachy_brain.speakable), transcribe with Whisper, check the
numbers came through. Runs ON THE LAPTOP (loads models):
    python -s tools/tts_number_check.py            # built-in phrases, PASS/FAIL each
    python -s tools/tts_number_check.py --as-is "Team one six seven eight won."   # speak exact text, show what was heard
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reachy_brain.speakable import speakable
from reachy_brain.stt import STT
from reachy_brain.tts import PocketTTS

# (text the model writes, the numbers a listener must hear, in order)
PHRASES = [
    ("Team 254 is ranked 3rd with a record of 9-1-0.", ["254", "9", "1", "0"]),
    ("Your next match is Qualification 12 at 3:30 pm.", ["12", "3", "30"]),
    ("Team 1678 won in 2025.", ["1678", "2025"]),
    ("Team 1678 won 120-98.", ["1678", "120", "98"]),
    ("Volume is now 15%.", ["15"]),
    ("Team 1005 is in pit 42.", ["1005", "42"]),
    ("We scored 128 points, up 12.5%.", ["128", "12", "5"]),
    ("The event has 3,600 teams.", ["3600"]),
    ("They won 120-98.", ["120", "98"]),
]
# Whisper sometimes writes small numbers as words ("nine wins, one loss"); homophones (won/to/for) are NOT counted
WORDNUM = {"zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7",
           "eight": "8", "nine": "9", "ten": "10"}
tts, stt = PocketTTS("alba", 4), STT("base.en", 4)


def hear(text):
    a = np.concatenate([c for c in tts.stream(text)])
    a16 = np.interp(np.linspace(0, len(a) - 1, int(len(a) * 16000 / 24000)), np.arange(len(a)), a).astype(np.float32)
    return stt.transcribe(a16).strip()


def numbers_in(heard):
    """Digits Whisper wrote, in order (thousands commas dropped). Spelled-out small words are kept as words."""
    h = re.sub(r"(?<=\d),(?=\d{3})", "", heard)
    return [WORDNUM.get(w.lower(), w) for w in re.findall(r"\d+|[A-Za-z]+", h) if w.isdigit() or w.lower() in WORDNUM]


def has_in_order(got, want):
    joined = " ".join(got)
    pos = 0
    for w in want:
        i = joined.find(w, pos)
        if i < 0:
            return False
        pos = i + len(w)
    return True


if len(sys.argv) > 2 and sys.argv[1] == "--as-is":
    for t in sys.argv[2:]:
        print(f"{t}  ->  {hear(t)}", flush=True)
    sys.exit(0)

fails = 0
for text, want in PHRASES:
    sp = speakable(text)
    heard = hear(sp)
    ok = has_in_order(numbers_in(heard), want)
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'}  {text}\n      spoken: {sp}\n      heard:  {heard}\n", flush=True)
print(f"{len(PHRASES) - fails}/{len(PHRASES)} passed")
