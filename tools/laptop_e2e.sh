#!/usr/bin/env bash
# Runs ON THE LAPTOP: brain end-to-end with the fake robot (WAVs as mic), then restarts the brain once to time the
# saved prompt-cache restore, then stops the brain.
# Usage: laptop_e2e.sh <frame.jpg>     Output: bench/e2e3/result.jsonl, brain.log, brain_restart.log, reply_*.wav
set -uo pipefail
L=/home/erikb/.config/Claude/scratch-workspaces/fa71b979-8948-49d6-a7e7-b78feff1391c/daaa5d4f-5b51-411e-979f-2fce8135e3a2/scratch-2026-11-06-72f273
R=$L/repo-desktop; PY=$L/bench/.venv/bin/python; OUT=$L/bench/e2e3; P=$L/bench/pipe
export HF_HOME=$L/bench/hf
mkdir -p "$OUT" "$P"
cd "$R" || exit 1
echo "== making extra test WAVs" >&2
"$PY" -s - "$P" <<'PYEOF'
import os, sys, numpy as np, soundfile as sf
sys.path.insert(0, ".")
from reachy_brain.tts import PocketTTS
t = PocketTTS("alba", 4)
for name, text in [("ghost", "You."), ("pause", "Please pause the music."), ("resume", "Please play the music."),
                   ("daftpunk", "Play some daft punk."), ("team", "Tell me about team 1678."),
                   ("nextmatch", "Tell me about our next match."), ("timer5", "Set a timer for five minutes."),
                   ("remind", "Remind me in twenty minutes to check the battery."),
                   ("remember", "Remember that our pit is A3."), ("recall", "What do you remember?"),
                   ("robotname", "What is our robot called?"), ("cancel", "Cancel my timers.")]:
    path = f"{sys.argv[1]}/user_{name}.wav"
    if not os.path.exists(path):
        sf.write(path, np.concatenate([c for c in t.stream(text)]), 24000)
PYEOF
wait_ready() { for _ in $(seq 1 120); do grep -q "brain ready" "$L/bench/live_brain.log" && return 0; sleep 3; done; return 1; }
"$R/tools/laptop_start_brain.sh" >&2
sleep 3; wait_ready || echo "brain not ready" >&2
echo "== brain ready, running fake robot" >&2
FRAME_JPEG=${1:-} LEAN=1 MEMORY="Our robot is named Zippy" "$PY" -s tools/fake_robot.py ws://127.0.0.1:8765 "$OUT" \
  "$P/user_joke.wav" "$P/user_ghost.wav" "$P/user_lofi.wav" "$P/user_pause.wav" "$P/user_resume.wav" \
  "$P/user_daftpunk.wav" "$P/user_weather.wav" "$P/user_team.wav" "$P/user_nextmatch.wav" "$P/user_happy.wav" \
  "$P/user_timer5.wav" "$P/user_remind.wav" "$P/user_remember.wav" "$P/user_recall.wav" "$P/user_robotname.wav" \
  "$P/user_cancel.wav" "$P/user_look.wav" "$P/user_worldseries.wav" > "$OUT/result.jsonl" 2> "$OUT/client.err"
echo "client exit=$?" >&2
cp "$L/bench/live_brain.log" "$OUT/brain.log"
echo "== restart: time the saved-cache restore" >&2
"$R/tools/laptop_start_brain.sh" >&2
sleep 3; wait_ready || echo "brain not ready after restart" >&2
FRAME_JPEG= LEAN=1 MEMORY="Our robot is named Zippy" "$PY" -s tools/fake_robot.py ws://127.0.0.1:8765 "$OUT/restart" \
  "$P/user_joke.wav" > "$OUT/restart.jsonl" 2>> "$OUT/client.err"
cp "$L/bench/live_brain.log" "$OUT/brain_restart.log"
"$R/tools/laptop_start_brain.sh" stop >&2
pgrep -x llama-server >/dev/null && echo "WARNING: llama-server still running" >&2
echo ALLDONE >&2
