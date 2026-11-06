"""QR scanning and FRC Nexus/Blue Alliance tools with fake HTTP and generated QR images."""
import time
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_gemini_dj import qr
from reachy_gemini_dj.frc import FRC


def qr_frame(text, size=480):
    qrcode = pytest.importorskip("qrcode")
    img = qrcode.make(text).convert("RGB").resize((300, 300))
    canvas = np.full((size, int(size * 4 / 3), 3), 90, np.uint8)       # code on a gray "camera" background
    canvas[90:390, 170:470] = np.asarray(img)
    return canvas[..., ::-1].copy()                                      # BGR like the camera


class Cfg(SimpleNamespace):
    def save(self): self.saved = getattr(self, "saved", 0) + 1


def cfg(**kw):
    base = dict(frc_team="", frc_event="", nexus_api_key="nk", tba_api_key="tk", nexus_link="", frc_alerts=True)
    return Cfg(**{**base, **kw})


def test_parse_frc_links():
    assert qr.parse_frc("https://frc.nexus/en/event/2026MOKC/team/1234") == {"source": "nexus", "event": "2026mokc", "team": "1234"}
    assert qr.parse_frc("https://www.thebluealliance.com/event/2026mokc")["event"] == "2026mokc"
    assert qr.parse_frc("https://www.thebluealliance.com/team/254")["team"] == "254"
    assert qr.parse_frc("https://frc.nexus/t/abc?event=2026txda&team=118") == {"source": "nexus", "event": "2026txda", "team": "118"}
    assert qr.parse_frc("hello world")["source"] == ""


def test_scan_reads_code_from_camera_and_saves_settings():
    c = cfg()
    frames = iter([None, np.zeros((480, 640, 3), np.uint8), qr_frame("https://frc.nexus/en/event/2026mokc/team/1234?k=SECRET")])
    f = FRC(c, get_frame=lambda: next(frames, None))
    res = f.scan_qr()
    assert res["team"] == "1234" and res["event"] == "2026mokc" and "SECRET" not in str(res)   # private link not returned
    assert c.nexus_link.endswith("SECRET") and c.saved == 1


def test_scan_non_frc_code_and_nothing_found():
    f = FRC(cfg(), get_frame=lambda: qr_frame("Hello robot"))
    assert f.scan_qr() == {"qr_text": "Hello robot"}
    f2 = FRC(cfg(), get_frame=lambda: np.zeros((10, 10, 3), np.uint8))
    qr_scan = qr.scan
    qr.scan = lambda gf, timeout_s=6.0, interval_s=0.15: qr_scan(gf, 0.3, 0.05)
    try:
        assert "could not find" in f2.scan_qr()["error"]
    finally:
        qr.scan = qr_scan


def nexus_event(now_ms, status="Now queuing", started=False):
    return {"eventKey": "2026mokc", "nowQueuing": "Qualification 12", "announcements": [{"announcement": "Lunch at noon"}],
            "matches": [
                {"label": "Qualification 3", "status": None, "redTeams": ["1234", "1", "2"], "blueTeams": ["3", "4", "5"],
                 "times": {"estimatedStartTime": now_ms - 3_600_000}},                      # long past: done
                {"label": "Qualification 12", "status": status, "redTeams": ["7", "8", "9"], "blueTeams": ["1234", "10", "11"],
                 "times": {"estimatedQueueTime": now_ms + 60_000, "estimatedStartTime": now_ms + 900_000,
                           **({"actualStartTime": now_ms} if started else {})}},
            ]}


def fake_http(routes):
    calls = []
    def get(url, headers, timeout=8.0):
        calls.append((url, headers))
        for k, v in routes.items():
            if url.endswith(k):
                return v() if callable(v) else v
        raise AssertionError("unexpected " + url)
    get.calls = calls
    return get


def test_next_match_from_nexus():
    now = time.time() * 1000
    http = fake_http({"/event/2026mokc": nexus_event(now)})
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=http)
    r = f.handle("frc_next_match", {})
    assert r["match"] == "Qualification 12" and r["alliance"] == "blue" and r["partners"] == ["10", "11"]
    assert r["status"] == "Now queuing" and r["start"].startswith("in about")
    assert http.calls[0][1]["Nexus-Api-Key"] == "nk"


def test_next_match_falls_back_to_tba_without_nexus_key():
    http = fake_http({"/team/frc1234/event/2026mokc/matches/simple": [
        {"comp_level": "qm", "match_number": 2, "set_number": 1, "actual_time": 1,
         "alliances": {"red": {"team_keys": ["frc1234"], "score": 50}, "blue": {"team_keys": ["frc9"], "score": 20}}},
        {"comp_level": "qm", "match_number": 9, "set_number": 1, "predicted_time": time.time() + 600,
         "alliances": {"red": {"team_keys": ["frc5", "frc6"], "score": -1}, "blue": {"team_keys": ["frc1234", "frc77"], "score": -1}}}]})
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc", nexus_api_key=""), http_get=http)
    r = f.handle("frc_next_match", {})
    assert r["match"] == "QM 9" and r["alliance"] == "blue" and r["partners"] == ["77"]
    assert http.calls[0][1]["X-TBA-Auth-Key"] == "tk"


def test_missing_settings_give_spoken_errors():
    f = FRC(cfg(nexus_api_key="", tba_api_key=""), http_get=fake_http({}))
    assert "Scan your Nexus QR" in f.handle("frc_event_status", {})["error"]
    f.config.frc_event = "2026mokc"
    assert "Nexus API key" in f.handle("frc_event_status", {})["error"]


def test_rankings_and_team_info():
    http = fake_http({
        "/event/2026mokc/rankings": {"rankings": [{"rank": 1, "team_key": "frc254"}, {"rank": 2, "team_key": "frc1234"}]},
        "/event/2026mokc/pits": {"254": "A3"},
        "/team/frc254/simple": {"nickname": "The Cheesy Poofs", "city": "San Jose", "state_prov": "CA"},
        "/team/frc254/event/2026mokc/status": {"qual": {"ranking": {"rank": 1, "record": {"wins": 9, "losses": 1, "ties": 0}}}},
    })
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=http)
    assert f.handle("frc_rankings", {"top": 2}) == {"top": [{"rank": 1, "team": "254"}, {"rank": 2, "team": "1234"}], "our_rank": 2}
    info = f.handle("frc_team_info", {"team": "254"})
    assert info["pit"] == "A3" and info["name"] == "The Cheesy Poofs" and info["record"] == "9-1-0"


def test_queuing_alerts_spoken_once_per_status():
    now = time.time() * 1000
    state = {"status": "Queuing soon"}
    http = fake_http({"/event/2026mokc": lambda: nexus_event(now, state["status"])})
    said = []
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), say=said.append, http_get=http)
    f.poll_once(); f.poll_once()
    state["status"] = "Now queuing"; f.poll_once()
    state["status"] = "On deck"; f.poll_once()
    assert said == ["Heads up! Qualification 12 is queuing soon. We are on the blue alliance.",
                    "Heads up! Qualification 12 is now queuing. We are on the blue alliance.",
                    "Heads up! Qualification 12 is on deck. We are on the blue alliance."]
    assert f.status["next_match"] == "Qualification 12 (On deck)" and f.status["now_queuing"] == "Qualification 12"
    f.config.frc_alerts = False
    state["status"] = "On field"; f.poll_once()
    assert len(said) == 3


def test_http_errors_become_readable():
    import urllib.error
    def boom(url, headers, timeout=8.0):
        raise urllib.error.HTTPError(url, 403, "Forbidden", {}, None)
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=boom)
    assert "rejected" in f.handle("frc_event_status", {})["error"]
