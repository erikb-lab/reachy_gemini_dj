#!/usr/bin/env bash
# Keep the brain laptop awake (Linux "caffeinate"): no suspend, hibernate or
# lid-close sleep, on AC or battery. Screen may still blank/lock - that is fine.
#
#   laptop_caffeinate.sh on      # keep awake (persists across reboots)
#   laptop_caffeinate.sh off     # restore normal sleep behaviour
#   laptop_caffeinate.sh status
#
# Needs sudo (passwordless on the laptop). Safe to run repeatedly.
set -uo pipefail

TARGETS="sleep.target suspend.target hibernate.target hybrid-sleep.target suspend-then-hibernate.target"
LOGIND_DROPIN=/etc/systemd/logind.conf.d/50-reachy-caffeinate.conf
KW=$(command -v kwriteconfig6 || command -v kwriteconfig5 || true)

die() { echo "ERROR: $*" >&2; exit 1; }

status() {
  echo "== sleep targets =="
  for t in $TARGETS; do printf '  %-32s %s\n' "$t" "$(systemctl is-enabled "$t" 2>/dev/null)"; done
  echo "== logind drop-in =="
  if [ -f "$LOGIND_DROPIN" ]; then sed 's/^/  /' "$LOGIND_DROPIN"; else echo "  (none)"; fi
  echo "== PowerDevil (KDE) =="
  for g in AC Battery LowBattery; do
    a=$(kreadconfig6 --file powerdevilrc --group "$g" --group SuspendAndShutdown --key AutoSuspendAction 2>/dev/null)
    l=$(kreadconfig6 --file powerdevilrc --group "$g" --group SuspendAndShutdown --key LidAction 2>/dev/null)
    echo "  $g: AutoSuspendAction=${a:-default} LidAction=${l:-default}"
  done
}

powerdevil_reload() {
  # Tell a running PowerDevil to re-read its config (only works inside the user session bus).
  qdbus6 org.kde.Solid.PowerManagement /org/kde/Solid/PowerManagement \
    org.kde.Solid.PowerManagement.refreshStatus >/dev/null 2>&1 || true
}

case "${1:-status}" in
  on)
    sudo -n true || die "sudo needs a password; run this from a terminal on the laptop"
    echo "Masking sleep targets..."
    sudo systemctl mask $TARGETS || die "systemctl mask failed"
    echo "Telling logind to ignore the lid and idle..."
    sudo mkdir -p /etc/systemd/logind.conf.d || die "mkdir failed"
    printf '[Login]\nHandleLidSwitch=ignore\nHandleLidSwitchExternalPower=ignore\nHandleLidSwitchDocked=ignore\nIdleAction=ignore\n' \
      | sudo tee "$LOGIND_DROPIN" >/dev/null || die "writing $LOGIND_DROPIN failed"
    # Reload (not restart) so the desktop session is not killed.
    sudo systemctl kill -s HUP systemd-logind || echo "  (logind reload failed; applies after reboot)"
    if [ -n "$KW" ]; then
      echo "Turning off KDE idle-suspend and lid action..."
      for g in AC Battery LowBattery; do
        "$KW" --file powerdevilrc --group "$g" --group SuspendAndShutdown --key AutoSuspendAction 0
        "$KW" --file powerdevilrc --group "$g" --group SuspendAndShutdown --key LidAction 0
      done
      powerdevil_reload
    fi
    status
    ;;
  off)
    sudo -n true || die "sudo needs a password; run this from a terminal on the laptop"
    sudo systemctl unmask $TARGETS || die "systemctl unmask failed"
    sudo rm -f "$LOGIND_DROPIN"
    sudo systemctl kill -s HUP systemd-logind || true
    if [ -n "$KW" ]; then
      for g in AC Battery LowBattery; do
        "$KW" --file powerdevilrc --group "$g" --group SuspendAndShutdown --key AutoSuspendAction --delete
        "$KW" --file powerdevilrc --group "$g" --group SuspendAndShutdown --key LidAction --delete
      done
      powerdevil_reload
    fi
    status
    ;;
  status) status ;;
  *) die "usage: $0 on|off|status" ;;
esac

# Undo: "laptop_caffeinate.sh off" unmasks the sleep targets, removes the logind
# drop-in and deletes the two PowerDevil keys it set (KDE defaults come back).
