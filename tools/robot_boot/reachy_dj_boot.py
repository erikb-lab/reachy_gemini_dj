#!/usr/bin/env python3
"""Runs ON THE ROBOT at boot: update Reachy Gemini DJ if the Hugging Face Space has a newer version, then start it.

Order of work (each step logs one line to the journal: journalctl -u reachy-dj-boot):
  1. wait for the Reachy daemon (http://127.0.0.1:8000) to report it is running and ready;
  2. unless updates are paused (``~/.reachy_dj_boot/no-update`` exists) or the robot has no internet, ask the daemon
     whether the installed app's Space commit differs from the latest one (``GET /api/apps/check-updates?force=true``);
  3. if it does: stop the app if it is running, back up the installed package, run the daemon's own update
     (``POST /api/apps/update/<app>``: uninstall + reinstall from the Space) and wait for it. If the update fails, the
     backup is put back so the robot still has a working app;
  4. start the app (``POST /api/apps/start-app/<app>``) unless it, or some other app, is already running.

It never fails the boot: any problem is logged and the script moves on to starting whatever version is installed.
Standard library only, so it runs with the robot's /usr/bin/python3. Safe to run by hand at any time:
    python3 ~/.reachy_dj_boot/reachy_dj_boot.py [--no-update] [--dry-run]
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import re
import shutil
import socket
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

APP = "reachy_gemini_dj"
DAEMON = "http://127.0.0.1:8000"
SITE = "/venvs/apps_venv/lib/python3.12/site-packages"
STATE = Path.home() / ".reachy_dj_boot"
PAUSE_FILE = STATE / "no-update"
BACKUP = STATE / "backup.tgz"

DAEMON_WAIT_S = 300
NET_WAIT_S = 90
UPDATE_WAIT_S = 20 * 60
START_WAIT_S = 90

log = logging.getLogger("reachy_dj_boot")


class Daemon:
    """The few Reachy daemon endpoints this needs. Tests swap it for a fake."""

    def __init__(self, base: str = DAEMON):
        self.base = base

    def _call(self, method: str, path: str, body=None, timeout: float = 30):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            raw = r.read()
        return json.loads(raw) if raw else None

    def status(self):
        return self._call("GET", "/api/daemon/status", timeout=5)

    def check_updates(self):
        return self._call("GET", "/api/apps/check-updates?force=true", timeout=180)

    def current_app(self):
        return self._call("GET", "/api/apps/current-app-status", timeout=10)

    def stop_app(self):
        return self._call("POST", "/api/apps/stop-current-app", timeout=60)

    def update(self, app: str):
        return self._call("POST", f"/api/apps/update/{app}", timeout=30)

    def job(self, job_id: str):
        return self._call("GET", f"/api/apps/job-status/{job_id}", timeout=10)

    def installed(self):
        return self._call("GET", "/api/apps/list-available/installed", timeout=60)

    def start(self, app: str):
        return self._call("POST", f"/api/apps/start-app/{app}", timeout=120)


def online(host: str = "huggingface.co", timeout: float = 5) -> bool:
    try:
        socket.create_connection((host, 443), timeout=timeout).close()
        return True
    except OSError:
        return False


def installed_version(site: str = SITE) -> str:
    try:
        text = Path(site, APP, "__init__.py").read_text()
        m = re.search(r'__version__\s*=\s*["\']([^"\']+)', text)
        return m.group(1) if m else "?"
    except OSError:
        return "not installed"


def backup(site: str = SITE, dest: Path = BACKUP) -> bool:
    paths = [p for p in [os.path.join(site, APP), *glob.glob(os.path.join(site, f"{APP}-*.dist-info"))] if os.path.exists(p)]
    if not paths:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")
    with tarfile.open(tmp, "w:gz") as tar:
        for p in paths:
            tar.add(p, arcname=os.path.basename(p), filter=lambda ti: None if "__pycache__" in ti.name else ti)
    tmp.replace(dest)
    return True


def restore(site: str = SITE, src: Path = BACKUP) -> bool:
    if not src.exists():
        return False
    for p in [os.path.join(site, APP), *glob.glob(os.path.join(site, f"{APP}-*.dist-info"))]:
        shutil.rmtree(p, ignore_errors=True)
    with tarfile.open(src, "r:gz") as tar:
        tar.extractall(site, filter="data")
    return True


def wait_until(check, timeout: float, every: float = 3, sleep=time.sleep, clock=time.monotonic):
    """Call check() until it returns something truthy or timeout passes; return the last value."""
    end = clock() + timeout
    while True:
        try:
            val = check()
        except Exception as e:          # daemon not up yet, connection refused, ...
            val = None
            log.debug("waiting: %s", e)
        if val or clock() >= end:
            return val
        sleep(every)


def daemon_ready(d: Daemon) -> bool:
    s = d.status() or {}
    return s.get("state") == "running" and bool((s.get("backend_status") or {}).get("ready"))


def running_app(d: Daemon):
    """(name, state) of the app the daemon is running, or (None, None)."""
    cur = d.current_app()
    if not cur:
        return None, None
    return (cur.get("info") or {}).get("name"), cur.get("state")


def update_available(d: Daemon):
    res = d.check_updates() or {}
    for a in res.get("apps_with_updates") or []:
        if a.get("app_name") == APP and a.get("update_available"):
            return a
    return None


def do_update(d: Daemon, info: dict, site: str, sleep=time.sleep, clock=time.monotonic) -> bool:
    before = installed_version(site)
    log.info("update found: Space commit %s -> %s (installed version %s)",
             (info.get("installed_sha") or "?")[:7], (info.get("latest_sha") or "?")[:7], before)
    name, state = running_app(d)
    if name == APP and state not in (None, "stopped", "error"):
        log.info("stopping the running app before updating")
        d.stop_app()
        wait_until(lambda: running_app(d)[0] != APP or running_app(d)[1] in ("stopped", "error"), 60, 2, sleep, clock)
    have_backup = backup(site)
    log.info("backed up the current install to %s" if have_backup else "nothing to back up (%s)", BACKUP)
    try:
        job_id = (d.update(APP) or {}).get("job_id")
        if not job_id:
            raise RuntimeError("the daemon did not start an update job")
        job = wait_until(lambda: (lambda j: j if j.get("status") in ("done", "failed") else None)(d.job(job_id)),
                         UPDATE_WAIT_S, 5, sleep, clock)
        if not job or job.get("status") != "done":
            tail = " | ".join((job or {}).get("logs", [])[-3:]) if job else "timed out"
            raise RuntimeError(f"update job ended {job.get('status') if job else 'without finishing'}: {tail}")
        apps = d.installed() or []
        if not any((a.get("name") if isinstance(a, dict) else None) == APP for a in apps):
            raise RuntimeError("the app is not listed as installed after the update")
    except Exception as e:
        log.error("update failed: %s", e)
        if have_backup and restore(site):
            log.warning("put the previous install back (version %s)", installed_version(site))
        return False
    log.info("updated: version %s -> %s", before, installed_version(site))
    return True


def start(d: Daemon, sleep=time.sleep, clock=time.monotonic) -> bool:
    name, state = running_app(d)
    if name == APP and state in ("starting", "running"):
        log.info("app already %s (version %s)", state, installed_version())
        return True
    if name and state in ("starting", "running"):
        log.info("another app (%s) is %s: leaving it alone", name, state)
        return False
    d.start(APP)
    ok = wait_until(lambda: running_app(d) == (APP, "running"), START_WAIT_S, 2, sleep, clock)
    if ok:
        log.info("app running (version %s)", installed_version())
    else:
        log.error("app did not reach 'running' within %d s: %s", START_WAIT_S, running_app(d))
    return bool(ok)


def run(d: Daemon, *, no_update: bool = False, dry_run: bool = False, site: str = SITE,
        net=online, sleep=time.sleep, clock=time.monotonic) -> dict:
    out = {"updated": False, "started": False, "skipped": None}
    if not wait_until(lambda: daemon_ready(d), DAEMON_WAIT_S, 3, sleep, clock):
        log.error("the Reachy daemon was not ready after %d s; giving up", DAEMON_WAIT_S)
        out["skipped"] = "daemon"
        return out
    log.info("daemon ready; installed version %s", installed_version(site))

    if no_update or PAUSE_FILE.exists():
        out["skipped"] = "paused"
        log.info("updates paused (%s); starting the installed version", PAUSE_FILE if PAUSE_FILE.exists() else "--no-update")
    elif not wait_until(net, NET_WAIT_S, 5, sleep, clock):
        out["skipped"] = "offline"
        log.warning("no internet after %d s; starting the installed version", NET_WAIT_S)
    else:
        try:
            info = update_available(d)
        except Exception as e:
            info = None
            out["skipped"] = "check-failed"
            log.warning("could not check for updates (%s); starting the installed version", e)
        if info is None and out["skipped"] is None:
            log.info("no new version on the Space")
        elif info and dry_run:
            log.info("dry run: would update %s -> %s", (info.get("installed_sha") or "?")[:7], (info.get("latest_sha") or "?")[:7])
        elif info:
            out["updated"] = do_update(d, info, site, sleep, clock)

    if dry_run:
        log.info("dry run: not starting the app")
        return out
    try:
        out["started"] = start(d, sleep, clock)
    except Exception as e:
        log.error("could not start the app: %s", e)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--no-update", action="store_true", help="skip the update check, just start the installed version")
    ap.add_argument("--dry-run", action="store_true", help="only report whether an update is available")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(Daemon(), no_update=a.no_update, dry_run=a.dry_run)
    return 0                             # never fail the boot


if __name__ == "__main__":
    raise SystemExit(main())
