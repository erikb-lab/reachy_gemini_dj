#!/usr/bin/env bash
# Runs ON THE LAPTOP: brain end-to-end with the fake robot (WAVs as mic), then stops the brain.
# Usage: laptop_e2e.sh <frame.jpg>     Output: bench/e2e2/result.jsonl, bench/e2e2/brain.log, reply_*.wav
set -uo pipefail
L=/home/erikb/.config/Claude/scratch-workspaces/fa71b979-8948-49d6-a7e7-b78feff1391c/daaa5d4f-5b51-411e-979f-2fce8135e3a2/scratch-2026-11-06-72f273
R=$L/repo-desktop; PY=$L/bench/.venv/bin/python; OUT=$L/bench/e2e2; P=$L/bench/pipe
export HF_HOME=$L/bench/hf
mkdir -p "$OUT" "$P"
cd "$R" || exit 1
echo "== making extra test WAVs" >&2
"$PY" -s - "$P" <<'PYEOF'
import sys, numpy as np, soundfile as sf
sys.path.insert(0, ".")
from reachy_brain.tts import PocketTTS
t = PocketTTS("alba", 4)
for name, text in [("ghost", "You."), ("pause", "Please pause the music."), ("resume", "Please play the music."),
                   ("daftpunk", "Play some daft punk."), ("team", "Tell me about team 1678.")]:
    a = np.concatenate([c for c in t.stream(text)])
    sf.write(f"{sys.argv[1]}/user_{name}.wav", a, 24000)
PYEOF
"$R/tools/laptop_start_brain.sh" >&2
sleep 3
for _ in $(seq 1 100); do grep -q "brain ready" "$L/bench/live_brain.log" && break; sleep 3; done
echo "== brain ready, running fake robot" >&2
FRAME_JPEG=${1:-} LEAN=1 "$PY" -s tools/fake_robot.py ws://127.0.0.1:8765 "$OUT" \
  "$P/user_joke.wav" "$P/user_ghost.wav" "$P/user_lofi.wav" "$P/user_pause.wav" "$P/user_resume.wav" \
  "$P/user_daftpunk.wav" "$P/user_weather.wav" "$P/user_team.wav" "$P/user_happy.wav" "$P/user_look.wav" \
  "$P/user_worldseries.wav" > "$OUT/result.jsonl" 2> "$OUT/client.err"
echo "client exit=$?" >&2
cp "$L/bench/live_brain.log" "$OUT/brain.log"
"$R/tools/laptop_start_brain.sh" stop >&2
pgrep -x llama-server >/dev/null && echo "WARNING: llama-server still running" >&2
echo ALLDONE >&2
