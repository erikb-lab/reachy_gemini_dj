#!/usr/bin/env bash
# Run ON THE LAPTOP: install the Reachy brain as a systemd user service (starts at login, restarts if it crashes).
#
# Usage:
#   install_brain_service.sh --python VENV_PYTHON --llama-server PATH --model GGUF --mmproj GGUF \
#                            [--repo DIR] [--port 8765] [--hf-home DIR] [--at-boot]
# Every path can also come from an environment variable: BRAIN_PYTHON, LLAMA_SERVER_BIN, BRAIN_MODEL, BRAIN_MMPROJ,
# BRAIN_REPO (default: the repo this script lives in), BRAIN_PORT, HF_HOME.
# --at-boot also enables lingering, so the brain starts at boot without anyone logging in (asks for sudo/polkit).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${BRAIN_PYTHON:-}"
LLAMA="${LLAMA_SERVER_BIN:-}"
MODEL="${BRAIN_MODEL:-}"
MMPROJ="${BRAIN_MMPROJ:-}"
REPO="${BRAIN_REPO:-$(dirname "$HERE")}"
PORT="${BRAIN_PORT:-8765}"
HFHOME="${HF_HOME:-}"
AT_BOOT=0

die() { echo "ERROR: $*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --python) PYTHON="${2:-}"; shift 2 ;;
    --llama-server) LLAMA="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --mmproj) MMPROJ="${2:-}"; shift 2 ;;
    --repo) REPO="${2:-}"; shift 2 ;;
    --port) PORT="${2:-}"; shift 2 ;;
    --hf-home) HFHOME="${2:-}"; shift 2 ;;
    --at-boot) AT_BOOT=1; shift ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "unknown option: $1 (try --help)" ;;
  esac
done

echo "== 1/6 Checking the inputs"
[ -n "$PYTHON" ] && [ -x "$PYTHON" ] || die "python venv interpreter not found or not executable: '${PYTHON}' (--python)"
[ -n "$LLAMA" ] && [ -x "$LLAMA" ] || die "llama-server not found or not executable: '${LLAMA}' (--llama-server)"
[ -n "$MODEL" ] && [ -f "$MODEL" ] || die "model file not found: '${MODEL}' (--model)"
[ -n "$MMPROJ" ] && [ -f "$MMPROJ" ] || die "mmproj file not found: '${MMPROJ}' (--mmproj)"
[ -d "$REPO/reachy_brain" ] || die "no reachy_brain/ folder in '${REPO}' (--repo)"
case "$PORT" in ''|*[!0-9]*) die "port must be a number: '${PORT}'" ;; esac
[ -f "$HERE/reachy-brain.service" ] || die "template missing: $HERE/reachy-brain.service"
command -v systemctl >/dev/null 2>&1 || die "systemctl not found; this needs systemd"
for p in "$PYTHON" "$LLAMA" "$MODEL" "$MMPROJ" "$REPO" "$HFHOME"; do
  case "$p" in *[[:space:]@\|\&]*) die "paths with spaces or @ | & are not supported here: '$p'" ;; esac
done
echo "   python:       $PYTHON"
echo "   llama-server: $LLAMA"
echo "   model:        $MODEL"
echo "   mmproj:       $MMPROJ"
echo "   repo:         $REPO"
echo "   port:         $PORT"

echo "== 2/6 Writing the unit file"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNIT="$UNIT_DIR/reachy-brain.service"
mkdir -p "$UNIT_DIR" || die "could not create $UNIT_DIR"
HF_LINE="# (HF_HOME not set: the default Hugging Face cache is used)"
[ -n "$HFHOME" ] && HF_LINE="Environment=HF_HOME=$HFHOME"
sed -e "s|@REPO@|$REPO|g" -e "s|@PYTHON@|$PYTHON|g" -e "s|@LLAMA_SERVER@|$LLAMA|g" \
    -e "s|@MODEL@|$MODEL|g" -e "s|@MMPROJ@|$MMPROJ|g" -e "s|@PORT@|$PORT|g" \
    -e "s|@HF_HOME_LINE@|$HF_LINE|" "$HERE/reachy-brain.service" > "$UNIT.tmp" || die "could not render the unit"
grep -q '@[A-Z_]*@' "$UNIT.tmp" && die "unfilled placeholder left in the unit (see $UNIT.tmp)"
mv "$UNIT.tmp" "$UNIT" || die "could not write $UNIT"
echo "   wrote $UNIT"

echo "== 3/6 Stopping any hand-started brain on port $PORT (so the service can take the port)"
if pgrep -f "reachy_brain[.]server" >/dev/null 2>&1 && ! systemctl --user is-active --quiet reachy-brain; then
  echo "   A reachy_brain.server is already running outside systemd. Stop it first, then run this again."
  echo "   (for example: pkill -TERM -f 'reachy_brain[.]server'; pkill -x llama-server)"
  exit 1
fi
echo "   nothing in the way"

echo "== 4/6 Reloading systemd and starting the service"
systemctl --user daemon-reload || die "systemctl --user daemon-reload failed"
systemctl --user enable --now reachy-brain || die "could not enable/start reachy-brain (journalctl --user -u reachy-brain)"

echo "== 5/6 Start at boot"
if [ "$AT_BOOT" = 1 ]; then
  loginctl enable-linger "$USER" || die "loginctl enable-linger failed"
  echo "   lingering enabled: the brain starts at boot, before anyone logs in"
else
  echo "   skipped (the brain starts when you log in). Re-run with --at-boot to start it at boot."
fi

echo "== 6/6 Status"
sleep 2
systemctl --user --no-pager status reachy-brain | head -n 12
echo
echo "Logs:    journalctl --user -u reachy-brain -f"
echo "Restart: systemctl --user restart reachy-brain"
echo "The first start can take a couple of minutes while the models load."

# UNDO:
#   systemctl --user disable --now reachy-brain
#   rm ~/.config/systemd/user/reachy-brain.service && systemctl --user daemon-reload
#   loginctl disable-linger "$USER"      # only if you used --at-boot
