"""Backend switching with fake sessions (no network, no models)."""
import time
from types import SimpleNamespace

import reachy_gemini_dj.backends as B


class FakeSess:
    def __init__(self, name):
        self.name, self.started, self.stopped = name, 0, 0
        self.state, self.last_error, self.motion = "idle", "", None
        self.connected, self.consecutive_failures, self.last_connect_ok = False, 0, 0.0
        self.turn_overdue, self.info, self.mic = False, {}, []

    def start(self): self.started += 1
    def stop(self): self.stopped += 1
    def init_client(self): return True
    def new_conversation(self): pass
    def push_mic(self, pcm): self.mic.append(pcm)
    def execute_tool(self, n, a): return {"ran": n}


def make(mode="auto", key="k"):
    cfg = SimpleNamespace(backend=mode, api_key=key, brain_url="ws://127.0.0.1:9")
    events = []
    vb = B.VoiceBackend(cfg, FakeSess("local"), FakeSess("gemini"), lambda k, t: events.append(t))
    return cfg, vb, events


def test_auto_starts_local_and_routes_mic():
    cfg, vb, _ = make()
    vb._activate(vb._preferred(), "startup")
    assert vb.active_name == "local" and vb.local.started == 1
    vb.push_mic(b"ab")
    assert vb.local.mic == [b"ab"] and vb.execute_tool("x", {}) == {"ran": "x"}


def test_fallback_after_repeated_failures(monkeypatch):
    cfg, vb, events = make()
    vb._activate(vb.local, "startup")
    vb.local.connected, vb.local.last_connect_ok = True, 1.0
    assert vb._local_failing() == ""
    vb.local.consecutive_failures = B.FAIL_LIMIT
    assert "failed" in vb._local_failing()
    monkeypatch.setattr(B, "RETRY_LOCAL_EVERY_S", 0.0)
    vb._stop.clear()
    # one watchdog step by hand
    why = vb._local_failing()
    vb._activate(vb.gemini, "fallback: " + why)
    assert vb.active_name == "gemini" and vb.local.stopped == 1 and vb.gemini.started == 1
    assert any("fallback" in e for e in events)


def test_unreachable_at_start_needs_grace_and_key():
    cfg, vb, _ = make(key="")
    vb._activate(vb.local, "startup")
    assert vb._local_failing() == ""                  # inside the grace window
    vb._started_at = time.monotonic() - B.START_GRACE_S - 1
    assert "unreachable" in vb._local_failing()


def test_manual_modes():
    cfg, vb, _ = make(mode="gemini")
    assert vb._preferred() is vb.gemini
    cfg.backend = "local"
    assert vb._preferred() is vb.local
    cfg.backend = "bogus"
    assert vb._mode() == "auto"


def test_watchdog_switches_back_when_brain_returns(monkeypatch):
    cfg, vb, _ = make()
    vb._activate(vb.gemini, "fallback: test")
    monkeypatch.setattr(B, "brain_reachable", lambda url, timeout=1.5: True)
    monkeypatch.setattr(B, "RETRY_LOCAL_EVERY_S", 0.0)
    vb._last_local_try = 0
    import threading
    t = threading.Thread(target=vb._watchdog, daemon=True)
    t.start()
    for _ in range(40):
        if vb.active_name == "local":
            break
        time.sleep(0.1)
    vb._stop.set()
    assert vb.active_name == "local"


def test_tool_specs_drop_unusable_tools():
    from reachy_gemini_dj.tool_schemas import robot_tool_specs
    names = lambda specs: {t["name"] for t in specs}
    allt = names(robot_tool_specs())
    lean = names(robot_tool_specs(google_services=set(), frc_enabled=False))
    assert {"calendar_events", "tasks_list", "gmail_check", "frc_rankings"} <= allt
    assert not lean & {"calendar_events", "calendar_add", "tasks_list", "tasks_add", "tasks_complete",
                       "gmail_check", "frc_next_match", "frc_rankings"}
    assert {"play_music", "scan_qr", "robot_pose"} <= lean
    assert "calendar_events" in names(robot_tool_specs(google_services={"calendar"}, frc_enabled=True))
    assert "tasks_list" not in names(robot_tool_specs(google_services={"calendar"}))


def test_hw_speaker_level_command(monkeypatch):
    from reachy_gemini_dj import audio
    calls = []
    monkeypatch.setattr(audio.shutil, "which", lambda n: "/usr/bin/amixer")

    class R:
        returncode, stderr = 0, ""
    monkeypatch.setattr(audio.subprocess, "run", lambda cmd, **k: calls.append(cmd) or R())
    assert audio.set_hw_speaker_level(100) == "speaker level 100%"
    assert calls[0][-3:] == ["sset", "PCM,0", "100%"]
    assert audio.set_hw_speaker_level(0) == "left alone" and len(calls) == 1
