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
    # what a real Nexus handout QR encodes (checked against the demo3528 event sheet; the key here is fake)
    assert qr.parse_frc("https://frc.nexus/demo3528/1300?key=FAKEKEY_abc-123") == {"source": "nexus", "event": "demo3528", "team": "1300"}
    assert qr.parse_frc("https://frc.nexus/2026mokc/254?key=x")["event"] == "2026mokc"
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
        "/team/frc254": {"nickname": "The Cheesy Poofs", "city": "San Jose", "state_prov": "CA"},
        "/team/frc254/events/simple": [], "/team/frc254/awards": [],
        "/team/frc254/event/2026mokc/status": {"qual": {"ranking": {"rank": 1, "record": {"wins": 9, "losses": 1, "ties": 0}}}},
    })
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=http)
    r = f.handle("frc_rankings", {"top": 2})
    assert {k: v for k, v in r.items() if k != "say"} == {"top": [{"rank": 1, "team": "254"}, {"rank": 2, "team": "1234"}], "our_rank": 2}
    info = f.handle("frc_team_info", {"team": "254"})
    assert info["pit"] == "A3" and info["name"] == "The Cheesy Poofs" and info["record_at_event"] == "9-1-0" and info["rank_at_event"] == 1


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


def test_scan_refuses_to_guess_between_several_codes():
    import numpy as np
    from reachy_gemini_dj import qr as _qr
    orig = _qr.decode
    _qr.decode = lambda frame: ["https://frc.nexus/demo3528/100?key=a", "https://frc.nexus/demo3528/1300?key=b"]
    try:
        with pytest.raises(_qr.MultipleCodes):
            _qr.scan(lambda: np.zeros((4, 4, 3), np.uint8), timeout_s=0.2, interval_s=0.05)
    finally:
        _qr.decode = orig


def test_team_info_works_without_an_event():
    calls = []

    def http(url, headers, timeout=8.0):
        calls.append(url)
        if url.endswith("/team/frc254"):
            return {"nickname": "The Cheesy Poofs", "city": "San Jose", "state_prov": "California", "rookie_year": 1999}
        if url.endswith("/events/simple"):
            return [{"key": "2025cmptx", "name": "Championship", "year": 2025, "start_date": "2025-04-16"},
                    {"key": "2025caph", "name": "Silicon Valley Regional", "year": 2025, "start_date": "2025-03-01"}]
        if url.endswith("/events/2025/statuses"):
            return {"2025caph": {"qual": {"ranking": {"rank": 2, "record": {"wins": 11, "losses": 1, "ties": 0}}},
                                 "playoff": {"level": "f"}}, "2025cmptx": None}
        if url.endswith("/awards"):
            return [{"name": "Regional Winner", "year": 2025}]
        raise AssertionError("unexpected " + url)

    f = FRC(cfg(frc_team="", frc_event=""), http_get=http)
    r = f.handle("frc_team_info", {"team": "254"})
    assert r["name"] == "The Cheesy Poofs" and r["rookie_year"] == 1999
    assert r["recent_events"][0] == "Championship (2025)"
    assert r["results_2025"][1] == {"event": "Silicon Valley Regional", "year": 2025, "qual_rank": 2,
                                    "record": "11-1-0", "playoffs": "f"} or r["results_2025"][0]["qual_rank"] == 2
    assert r["recent_awards"] == ["Regional Winner (2025)"]
    assert "pit" not in r and not any("nexus" in c for c in calls)
    assert "error" not in r


# ---------------- TBA cache ----------------
from reachy_gemini_dj import frc as frc_mod


class Clock:
    t = 1000.0

    def __call__(self):
        return self.t


def cached_frc(routes, clock=None, **kw):
    """FRC with a conditional getter: routes[url_suffix] -> (body, etag); honours If-None-Match."""
    calls = []

    def cond(url, headers):
        calls.append((url, dict(headers)))
        for k, v in routes.items():
            if url.endswith(k):
                body, etag = v() if callable(v) else v
                if etag and headers.get("If-None-Match") == etag:
                    return frc_mod.NOT_MODIFIED, None
                return body, etag
        raise AssertionError("unexpected " + url)
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc", **kw), http_get=lambda *a, **k: None, http_get_cond=cond)
    if clock:
        f._cache = frc_mod.TTLCache(clock=clock)
    f.calls = calls
    return f


def test_cache_hit_ttl_and_304_revalidation():
    clk = Clock()
    f = cached_frc({"/team/frc254": ({"nickname": "X"}, 'W/"e1"'), "/event/2026mokc/rankings": ({"rankings": []}, 'W/"r1"')}, clk)
    assert f._tba("/team/frc254") == {"nickname": "X"}
    assert f._tba("/team/frc254") == {"nickname": "X"} and len(f.calls) == 1        # fresh: no request
    clk.t += 3600 * 23
    f._tba("/team/frc254")
    assert len(f.calls) == 1                                                          # profile TTL is 24 h
    clk.t += 3600 * 2
    assert f._tba("/team/frc254") == {"nickname": "X"} and len(f.calls) == 2
    assert f.calls[1][1]["If-None-Match"] == 'W/"e1"'                                 # revalidated, 304 reused body
    f._tba("/team/frc254")
    assert len(f.calls) == 2                                                          # and the clock restarted
    # current-event data lives 60 s
    f._tba("/event/2026mokc/rankings")
    f._tba("/event/2026mokc/rankings")
    n = len(f.calls)
    clk.t += 59
    f._tba("/event/2026mokc/rankings")
    assert len(f.calls) == n
    clk.t += 2
    f._tba("/event/2026mokc/rankings")
    assert len(f.calls) == n + 1


def test_cache_changed_body_replaces_entry():
    clk = Clock()
    state = {"v": ({"n": 1}, '"a"')}
    f = cached_frc({"/team/frc5": lambda: state["v"]}, clk)
    assert f._tba("/team/frc5") == {"n": 1}
    state["v"] = ({"n": 2}, '"b"')
    clk.t += 25 * 3600
    assert f._tba("/team/frc5") == {"n": 2}


def test_cache_ttl_by_kind():
    f = cached_frc({})
    assert f._ttl("/team/frc254") == 24 * 3600
    assert f._ttl("/team/frc254/events/simple") == 3600 and f._ttl("/team/frc254/awards") == 3600
    assert f._ttl("/team/frc254/events/2025/statuses") == 3600
    assert f._ttl("/event/2026mokc/teams/statuses") == 60 and f._ttl("/team/frc254/event/2026mokc/status") == 60
    assert f._ttl("/event/2026mokc/rankings") == 60 and f._ttl("/event/2025cmptx/rankings") == 3600


def test_cache_does_not_store_errors_or_nexus_and_is_bounded():
    import urllib.error
    clk = Clock()
    state = {"fail": True}

    def route():
        if state["fail"]:
            raise urllib.error.HTTPError("u", 500, "boom", {}, None)
        return ({"ok": 1}, None)
    f = cached_frc({"/team/frc9": route}, clk)
    with pytest.raises(urllib.error.HTTPError):
        f._tba("/team/frc9")
    state["fail"] = False
    assert f._tba("/team/frc9") == {"ok": 1} and len(f.calls) == 2                    # error was not cached
    # Nexus goes straight to the plain getter every time
    n = []
    g = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=lambda u, h, t=8.0: n.append(u) or {"matches": []})
    g._nexus("/event/2026mokc")
    g._nexus("/event/2026mokc")
    assert len(n) == 2
    c = frc_mod.TTLCache(max_entries=3)
    for i in range(5):
        c.fetch(f"u{i}", 60, {}, lambda u, h: ({"i": u}, None))
    assert len(c) == 3 and "u0" not in c._d and "u4" in c._d


def test_default_getter_sends_etag_and_handles_304(monkeypatch):
    import io
    import urllib.error

    class Resp(io.BytesIO):
        headers = {"ETag": '"abc"'}

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_open(req, timeout=0):
        if req.get_header("If-none-match") == '"abc"':
            raise urllib.error.HTTPError(req.full_url, 304, "Not Modified", {}, None)
        return Resp(b'{"a": 1}')
    monkeypatch.setattr(frc_mod.urllib.request, "urlopen", fake_open)
    assert frc_mod._get_cond("http://x", {}) == ({"a": 1}, '"abc"')
    assert frc_mod._get_cond("http://x", {"If-None-Match": '"abc"'}) == (frc_mod.NOT_MODIFIED, None)


# ---------------- briefing + spoken summaries ----------------
def match_routes(now_ms):
    return {
        "/event/2026mokc/teams/simple": [{"team_number": n, "nickname": nm} for n, nm in
                                         [(1234, "Us"), (10, "The X"), (11, "Eleven"), (7, "Seven"), (8, "Eight"), (9, "Nine")]],
        "/event/2026mokc/teams/statuses": {
            "frc10": {"qual": {"ranking": {"rank": 3, "record": {"wins": 5, "losses": 2, "ties": 0}}}},
            "frc11": {"qual": {"ranking": {"rank": 12, "record": {"wins": 2, "losses": 5, "ties": 0}}}},
            "frc7": {"qual": {"ranking": {"rank": 1, "record": {"wins": 7, "losses": 0, "ties": 0}}}},
            "frc8": None},
        "/event/2026mokc": nexus_event(now_ms),
    }


def test_next_match_briefing_has_opponents_and_team_details():
    http = fake_http(match_routes(time.time() * 1000))
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=http)
    r = f.handle("frc_next_match", {})
    assert r["match"] == "Qualification 12" and r["partners"] == ["10", "11"] and r["opponents"] == ["7", "8", "9"]
    assert r["partner_details"][0] == {"team": "10", "name": "The X", "rank": 3, "record": "5-2-0"}
    assert r["opponent_details"][1] == {"team": "8", "name": "Eight"}                  # no status yet
    assert r["opponent_details"][0]["record"] == "7-0-0"
    assert len([u for u, _ in http.calls if u.endswith("/teams/statuses")]) == 1       # one call covers every team
    say = r["say"]
    assert say.startswith("Our next match is Qualification 12, in about 15 minutes.")
    assert "We're on the blue alliance with team 10, The X, 5-2-0, ranked third and team 11" in say
    assert "Against team 7, Seven, 7-0-0, ranked first" in say and len(say.split()) <= 60
    print("SAY next_match:", say)


def test_next_match_briefing_without_tba_key():
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc", tba_api_key=""),
            http_get=fake_http({"/event/2026mokc": nexus_event(time.time() * 1000)}))
    r = f.handle("frc_next_match", {})
    assert "error" not in r and r["partner_details"] == [{"team": "10"}, {"team": "11"}]
    assert r["say"] == ("Our next match is Qualification 12, in about 15 minutes. It's queuing now. "
                        "We're on the blue alliance with team 10 and team 11. Against team 7, team 8 and team 9.")


def test_next_match_say_is_trimmed_when_long():
    routes = match_routes(time.time() * 1000)
    long_names = "The Extraordinarily Long Named Robotics Collective Of Somewhere"
    routes["/event/2026mokc/teams/simple"] = [{"team_number": n, "nickname": long_names} for n in (10, 11, 7, 8, 9)]
    routes["/event/2026mokc/teams/statuses"] = {f"frc{n}": {"qual": {"ranking": {"rank": n, "record": {"wins": 1, "losses": 1, "ties": 0}}}}
                                                for n in (10, 11, 7, 8, 9)}
    r = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=fake_http(routes)).handle("frc_next_match", {})
    assert len(r["say"].split()) <= 60 and "team 9" in r["say"]


def test_next_match_tba_fallback_includes_opponents_and_say():
    http = fake_http({"/team/frc1234/event/2026mokc/matches/simple": [
        {"comp_level": "qm", "match_number": 9, "set_number": 1, "predicted_time": time.time() + 600,
         "alliances": {"red": {"team_keys": ["frc5", "frc6"], "score": -1}, "blue": {"team_keys": ["frc1234", "frc77"], "score": -1}}}]})
    r = FRC(cfg(frc_team="1234", frc_event="2026mokc", nexus_api_key=""), http_get=http).handle("frc_next_match", {})
    assert r["opponents"] == ["5", "6"] and "Against team 5 and team 6" in r["say"] and "with team 77" in r["say"]


def test_team_info_say():
    def http(url, headers, timeout=8.0):
        if url.endswith("/team/frc254"):
            return {"nickname": "The Cheesy Poofs", "city": "San Jose", "state_prov": "California", "rookie_year": 1999}
        if url.endswith("/events/simple"):
            return [{"key": "2025caph", "name": "Silicon Valley Regional", "year": 2025, "start_date": "2025-03-01"}]
        if url.endswith("/events/2025/statuses"):
            return {"2025caph": {"qual": {"ranking": {"rank": 2, "record": {"wins": 11, "losses": 1, "ties": 0}}},
                                 "playoff": {"level": "sf", "status": "eliminated"}}}
        if url.endswith("/awards"):
            return [{"name": "Regional Winner", "year": 2025}, {"name": "Engineering Inspiration", "year": 2024}, {"name": "Old", "year": 2010}]
        raise AssertionError(url)
    r = FRC(cfg(frc_team="", frc_event=""), http_get=http).handle("frc_team_info", {"team": "254"})
    assert r["say"] == ("Team 254 is The Cheesy Poofs, from San Jose, California. They started in 1999. "
                        "In 2025 at Silicon Valley Regional they ranked second with a record of 11-1-0 and made it to the semifinals. "
                        "Recent awards: Regional Winner in 2025 and Engineering Inspiration in 2024.")
    print("SAY team_info:", r["say"])
    won = frc_mod.FRC._say_team({"team": "1", "name": "N"}, [{"event": "E", "year": 2025, "qual_rank": 1, "record": "9-0-0",
                                                             "playoffs": "f", "playoff_status": "won"}])
    assert won.endswith("ranked first with a record of 9-0-0 and won the event.")
    for lvl, words in (("ef", "the eighth-finals"), ("qf", "the quarterfinals"), ("f", "the finals")):
        assert words in frc_mod.FRC._say_team({"team": "1", "name": "N"}, [{"event": "E", "year": 1, "playoffs": lvl}])


def test_rankings_and_status_say_and_no_secrets():
    routes = {"/event/2026mokc/rankings": {"rankings": [
        {"rank": 1, "team_key": "frc254", "record": {"wins": 9, "losses": 1, "ties": 0}},
        {"rank": 2, "team_key": "frc1234", "record": {"wins": 8, "losses": 2, "ties": 0}}]},
        "/event/2026mokc": {"nowQueuing": "Qualification 12",
                            "announcements": [{"announcement": "Lunch at noon"},
                                              {"announcement": "Watch https://frc.nexus/x?key=SECRETKEY"}]}}
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc", nexus_link="https://frc.nexus/2026mokc/1234?key=SECRETKEY"),
            http_get=fake_http(routes))
    rk = f.handle("frc_rankings", {"top": 2})
    assert rk["say"] == "At the top we have team 254 in first and team 1234 in second. We're in second place, 8-2-0."
    st = f.handle("frc_event_status", {})
    assert st["say"] == "Qualification 12 is queuing now. Latest announcement: Lunch at noon"   # link announcement skipped
    for r in (rk, st):
        assert "SECRET" not in r["say"] and "http" not in r["say"] and "frc.nexus" not in r["say"]
    print("SAY rankings:", rk["say"])
    print("SAY event_status:", st["say"])


def test_errors_have_no_say():
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc", tba_api_key="", nexus_api_key=""), http_get=fake_http({}))
    for tool, args in (("frc_rankings", {}), ("frc_event_status", {}), ("frc_next_match", {}), ("frc_team_info", {"team": "x"})):
        r = f.handle(tool, args)
        assert "error" in r and "say" not in r
