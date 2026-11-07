#!/usr/bin/env bash
# Runs ON THE LAPTOP: start llama-server with several -t/-tb settings and run tools/bench_llm.py against each.
# Usage: laptop_bench_threads.sh <repo dir> <out.jsonl> "t:tb[:extra]" ...
set -uo pipefail
L=/home/erikb/.config/Claude/scratch-workspaces/fa71b979-8948-49d6-a7e7-b78feff1391c/daaa5d4f-5b51-411e-979f-2fce8135e3a2/scratch-2026-11-06-72f273
REPO=$1; OUT=$2; shift 2
BIN=$L/llama-cpp-b12110/llama-b12110/llama-server
MODEL=$L/models/groot/GrootTeam_Groot-19B-A3B-261023-Q4_K_M.gguf
PY=$L/bench/.venv/bin/python
for cfg in "$@"; do
  IFS=: read -r T TB EXTRA <<<"$cfg"
  echo "== -t $T -tb $TB $EXTRA" >&2
  # shellcheck disable=SC2086
  "$BIN" -m "$MODEL" --alias brain --host 127.0.0.1 --port 8099 -t "$T" -tb "$TB" -c 6144 -np 1 \
      --reasoning off --no-webui $EXTRA > "$OUT.server.log" 2>&1 &
  SP=$!
  for _ in $(seq 1 200); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; kill -0 $SP 2>/dev/null || break; sleep 1.5; done
  (cd "$REPO" && "$PY" -s tools/bench_llm.py http://127.0.0.1:8099 "t$T-tb$TB$EXTRA") >> "$OUT" || echo "bench failed for $cfg" >&2
  kill $SP; wait $SP 2>/dev/null
done
pgrep -x llama-server && echo "WARNING: llama-server still running" >&2
echo DONE >&2
