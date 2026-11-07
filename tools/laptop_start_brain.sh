#!/usr/bin/env bash
# Runs ON THE LAPTOP: (re)start the brain server detached, logging to bench/live_brain.log.
# Usage: laptop_start_brain.sh [stop]
L=/home/erikb/.config/Claude/scratch-workspaces/fa71b979-8948-49d6-a7e7-b78feff1391c/daaa5d4f-5b51-411e-979f-2fce8135e3a2/scratch-2026-11-06-72f273
pkill -TERM -f "python -s -m reachy_brain[.]server"
for _ in $(seq 1 20); do pgrep -x llama-server >/dev/null || break; sleep 1; done
pkill -x llama-server 2>/dev/null
[ "${1:-}" = stop ] && { echo stopped; exit 0; }
cd "$L/repo-desktop" || exit 1
HF_HOME=$L/bench/hf nohup "$L/bench/.venv/bin/python" -s -m reachy_brain.server \
  --llama-server-bin "$L/llama-cpp-b12110/llama-b12110/llama-server" \
  --model "$L/models/groot/GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf" --mmproj "$L/models/groot/mmproj-f16.gguf" \
  --port 8765 > "$L/bench/live_brain.log" 2>&1 < /dev/null &
disown
echo started
