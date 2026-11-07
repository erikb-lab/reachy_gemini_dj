"""tools/robot_boot/reachy_dj_boot.py: update-if-newer-then-start, against a fake Reachy daemon."""
import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "reachy_dj_boot", Path(__file__).resolve().parents[1] / "tools" / "robot_boot" / "reachy_dj_boot.py")
boot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boot)

APP = boot.APP


class FakeDaemon:
    def __init__(self, site, update=None, job_status="done", running=None, ready_after=0, check_error=None,
                 new_version="0.8.0"):
        self.site, self.upd, self.job_status, self.new_version = site, update, job_status, new_version
        self.app = running                   # (name, state) or None
        self.ready_after, self.check_error = ready_after, check_error
        self.calls = []

    def status(self):
        self.calls.append("status")
        if self.ready_after > 0:
            self.ready_after -= 1
            raise ConnectionRefusedError("daemon still starting")
        return {"state": "running", "backend_status": {"ready": True}}

    def check_updates(self):
        self.calls.append("check")
        if self.check_error:
            raise self.check_error
        return {"apps_with_updates": [self.upd] if self.upd else [], "apps_checked": 1}

    def current_app(self):
        return None if not self.app else {"info": {"name": self.app[0]}, "state": self.app[1]}

    def stop_app(self):
        self.calls.append("stop")
        self.app = None

    def update(self, app):
        self.calls.append("update")
        pkg = Path(self.site, APP)
        if self.job_status == "done":
            (pkg / "__init__.py").write_text(f'__version__ = "{self.new_version}"\n')
        else:                                # failed half way: the daemon uninstalled it first
            for p in Path(self.site).glob(f"{APP}*"):
                import shutil
                shutil.rmtree(p)
        return {"job_id": "j1"}

    def job(self, job_id):
        return {"command": "update", "status": self.job_status, "logs": ["pip: network unreachable"]}

    def installed(self):
        return [{"name": APP}] if Path(self.site, APP).exists() else []

    def start(self, app):
        self.calls.append("start")
        self.app = (app, "running")


@pytest.fixture
def site(tmp_path, monkeypatch):
    s = tmp_path / "site-packages"
    (s / APP).mkdir(parents=True)
    (s / APP / "__init__.py").write_text('__version__ = "0.7.0"\n')
    (s / f"{APP}-0.7.0.dist-info").mkdir()
    (s / f"{APP}-0.7.0.dist-info" / "METADATA").write_text("Name: reachy_gemini_dj\n")
    monkeypatch.setattr(boot, "STATE", tmp_path / "state")
    monkeypatch.setattr(boot, "PAUSE_FILE", tmp_path / "state" / "no-update")
    monkeypatch.setattr(boot, "BACKUP", tmp_path / "state" / "backup.tgz")
    monkeypatch.setattr(boot, "SITE", str(s))
    return s


class Clock:
    def __init__(self): self.t = 0.0
    def __call__(self): return self.t
    def sleep(self, s): self.t += s


def go(d, site, **kw):
    c = Clock()
    kw.setdefault("net", lambda: True)
    return boot.run(d, site=str(site), sleep=c.sleep, clock=c, **kw)


UPD = {"app_name": APP, "update_available": True, "installed_sha": "d3531b7aaa", "latest_sha": "362034cbbb"}


def test_no_update_just_starts(site):
    d = FakeDaemon(site, ready_after=3)
    out = go(d, site)
    assert out == {"updated": False, "started": True, "skipped": None}
    assert "update" not in d.calls and d.calls[-1] == "start"


def test_update_then_start_new_version(site):
    d = FakeDaemon(site, update=UPD, running=(APP, "running"))
    out = go(d, site)
    assert out["updated"] and out["started"]
    assert d.calls.index("stop") < d.calls.index("update") < d.calls.index("start")
    assert boot.installed_version(str(site)) == "0.8.0"


def test_failed_update_restores_previous_install_and_starts_it(site):
    d = FakeDaemon(site, update=UPD, job_status="failed")
    out = go(d, site)
    assert not out["updated"] and out["started"]
    assert boot.installed_version(str(site)) == "0.7.0"
    assert (site / f"{APP}-0.7.0.dist-info" / "METADATA").exists()


def test_offline_skips_check_and_starts(site):
    d = FakeDaemon(site, update=UPD)
    out = go(d, site, net=lambda: False)
    assert out["skipped"] == "offline" and out["started"] and "check" not in d.calls


def test_pause_file_skips_update(site):
    boot.PAUSE_FILE.parent.mkdir(parents=True)
    boot.PAUSE_FILE.touch()
    d = FakeDaemon(site, update=UPD)
    out = go(d, site)
    assert out["skipped"] == "paused" and "check" not in d.calls and out["started"]


def test_check_error_still_starts(site):
    d = FakeDaemon(site, check_error=TimeoutError("hf slow"))
    out = go(d, site)
    assert out["skipped"] == "check-failed" and out["started"]


def test_leaves_other_running_app_alone(site):
    d = FakeDaemon(site, running=("reachy_companion", "running"))
    out = go(d, site)
    assert not out["started"] and "start" not in d.calls


def test_dry_run_changes_nothing(site):
    d = FakeDaemon(site, update=UPD)
    out = go(d, site, dry_run=True)
    assert not out["updated"] and not out["started"] and "update" not in d.calls and "start" not in d.calls


def test_daemon_never_ready_gives_up(site):
    d = FakeDaemon(site, ready_after=10 ** 6)
    out = go(d, site)
    assert out["skipped"] == "daemon" and "start" not in d.calls
