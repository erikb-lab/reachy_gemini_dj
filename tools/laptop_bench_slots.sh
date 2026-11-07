#!/usr/bin/env bash
# Runs ON THE LAPTOP: llama-server with 2 slots + mmproj, then tools/bench_slots.py. Args: extra llama-server flags
L=/home/erikb/.config/Claude/scratch-workspaces/fa71b979-8948-49d6-a7e7-b78feff1391c/daaa5d4f-5b51-411e-979f-2fce8135e3a2/scratch-2026-11-06-72f273
"$L/llama-cpp-b12110/llama-b12110/llama-server" -m "$L/models/groot/GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf" \
  --mmproj "$L/models/groot/mmproj-f16.gguf" --alias brain --host 127.0.0.1 --port 8099 -t 4 -c 12288 -np 2 \
  --reasoning off --no-webui --slots "$@" > "$L/bench/slots_server.log" 2>&1 &
SP=$!
for _ in $(seq 1 100); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; sleep 1.5; done
cd "$L/repo-desktop" && "$L/bench/.venv/bin/python" -s tools/bench_slots.py http://127.0.0.1:8099 "$L/bench/frame_team1678.jpg"
kill $SP; wait $SP 2>/dev/null
grep -iE "slot|cache" "$L/bench/slots_server.log" | grep -iE "selected|restor|evict|reuse|id_slot|unified" | tail -8
