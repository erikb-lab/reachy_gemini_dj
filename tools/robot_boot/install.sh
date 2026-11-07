#!/usr/bin/env bash
# Run ON THE DESKTOP (or any machine with ssh access to the robot): install the boot service on Reachy Mini.
# At every boot it updates Reachy Gemini DJ from its Hugging Face Space if there is a newer version, then starts it.
# Asks for the robot's sudo password once (to put the unit in /etc/systemd/system and enable it).
#
# Usage: tools/robot_boot/install.sh [user@host]      (default pollen@reachy-mini.local)
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT="${1:-pollen@reachy-mini.local}"
die() { echo "ERROR: $*" >&2; exit 1; }

echo "== 1/4 Checking the robot answers at $ROBOT"
ssh -o ConnectTimeout=8 "$ROBOT" true || die "cannot ssh to $ROBOT (is it booted and on the network?)"

echo "== 2/4 Copying the boot script to ~/.reachy_dj_boot/ (no sudo needed)"
ssh "$ROBOT" 'mkdir -p ~/.reachy_dj_boot' || die "could not create ~/.reachy_dj_boot"
scp -q "$HERE/reachy_dj_boot.py" "$HERE/reachy-dj-boot.service" "$ROBOT":.reachy_dj_boot/ || die "copy failed"

echo "== 3/4 Installing and enabling the systemd unit (type the robot's sudo password when asked)"
ssh -t "$ROBOT" 'sudo install -m 644 ~/.reachy_dj_boot/reachy-dj-boot.service /etc/systemd/system/reachy-dj-boot.service \
  && sudo systemctl daemon-reload && sudo systemctl enable reachy-dj-boot.service' || die "installing the unit failed"

echo "== 4/4 Checking"
ssh "$ROBOT" 'systemctl is-enabled reachy-dj-boot.service; python3 ~/.reachy_dj_boot/reachy_dj_boot.py --dry-run' \
  || die "check failed"
echo
echo "OK: the robot will update-if-newer and start Reachy Gemini DJ on every boot."
echo "Logs after a boot:  ssh $ROBOT journalctl -u reachy-dj-boot -b"
echo "Pause updates:      ssh $ROBOT touch ~/.reachy_dj_boot/no-update   (delete the file to resume)"

# Undo:
#   ssh -t pollen@reachy-mini.local 'sudo systemctl disable reachy-dj-boot.service && \
#     sudo rm /etc/systemd/system/reachy-dj-boot.service && sudo systemctl daemon-reload && rm -rf ~/.reachy_dj_boot'
