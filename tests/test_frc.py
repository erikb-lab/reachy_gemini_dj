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
                        "In 2025 at Silicon Valley Regional they ranked 2nd with a record of 11-1-0 and made it to the semifinals. "
                        "Recent awards: Regional Winner in 2025 and Engineering Inspiration in 2024.")
    print("SAY team_info:", r["say"])
    won = frc_mod.FRC._say_team({"team": "1", "name": "N"}, [{"event": "E", "year": 2025, "qual_rank": 1, "record": "9-0-0",
                                                             "playoffs": "f", "playoff_status": "won"}])
    assert won.endswith("ranked 1st with a record of 9-0-0 and won the event.")
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


# ---------------- team lookups by event / season / career ----------------
EV = [  # a team's events across years (TBA /events/simple shape)
    {"key": "2026mndu", "name": "Lake Superior Regional", "city": "Duluth", "state_prov": "Minnesota", "year": 2026,
     "event_type": 0, "start_date": "2026-03-20"},
    {"key": "2026mnmi", "name": "Minnesota 10000 Lakes Regional presented by Foo", "city": "Minneapolis",
     "state_prov": "Minnesota", "year": 2026, "event_type": 0, "start_date": "2026-04-03"},
    {"key": "2026hop", "name": "Hopper Division", "city": "Houston", "state_prov": "Texas", "year": 2026,
     "event_type": 3, "start_date": "2026-04-15"},
    {"key": "2026cmptx", "name": "Einstein Field", "city": "Houston", "state_prov": "Texas", "year": 2026,
     "event_type": 4, "start_date": "2026-04-18"},
    {"key": "2025mndu", "name": "Lake Superior Regional", "city": "Duluth", "state_prov": "Minnesota", "year": 2025,
     "event_type": 0, "start_date": "2025-03-21"},
    {"key": "2025mndu2", "name": "Northern Lights Regional", "city": "Duluth", "state_prov": "Minnesota", "year": 2025,
     "event_type": 0, "start_date": "2025-04-04"},
    {"key": "2019mndu", "name": "Lake Superior Regional", "city": "Duluth", "state_prov": "Minnesota", "year": 2019,
     "event_type": 0, "start_date": "2019-03-20"},
    {"key": "2019mnmi", "name": "Minnesota 10000 Lakes Regional", "city": "Minneapolis", "state_prov": "Minnesota",
     "year": 2019, "event_type": 0, "start_date": "2019-04-03"},
    {"key": "2019arc", "name": "Archimedes Division", "city": "Houston", "state_prov": "Texas", "year": 2019,
     "event_type": 3, "start_date": "2019-04-17"},
]


def status(rank=None, of=60, w=0, l=0, t=0, alliance=None, pick=None, level=None, pst=None, done=True):
    st = {"qual": {"num_teams": of, "status": "completed" if done else "playing",
                   "ranking": {"rank": rank, "record": {"wins": w, "losses": l, "ties": t}} if rank else None}}
    if alliance:
        st["alliance"] = {"number": alliance, "pick": pick}
    if level:
        st["playoff"] = {"level": level, "status": pst, "record": {"wins": 3, "losses": 1, "ties": 0}}
    return st


ST_2026 = {"2026mndu": status(4, 60, 9, 3, 0, 2, 1, "sf", "eliminated"),
           "2026mnmi": status(1, 58, 11, 1, 0, 1, 0, "f", "won"),
           "2026hop": status(30, 75, 6, 6, 0),
           "2026cmptx": None}
ST_2019 = {"2019mndu": status(9, 50, 8, 4, 0, 3, 2, "qf", "eliminated"), "2019mnmi": status(20, 60, 6, 6, 0),
           "2019arc": status(40, 75, 5, 7, 0)}
AWARDS = [
    {"name": "Quality Award sponsored by Motorola", "award_type": 17, "year": 2026, "event_key": "2026mndu"},
    {"name": "Regional Winners", "award_type": 1, "year": 2026, "event_key": "2026mnmi"},
    {"name": "Regional Finalists", "award_type": 2, "year": 2025, "event_key": "2025mndu"},
    {"name": "Regional Winners", "award_type": 1, "year": 2019, "event_key": "2019mndu"},
    {"name": "Championship Winners", "award_type": 1, "year": 2026, "event_key": "2026cmptx"},
    {"name": "Impact Award", "award_type": 69, "year": 2025, "event_key": "2025mndu"},
    {"name": "Engineering Inspiration Award", "award_type": 9, "year": 2019, "event_key": "2019mndu"},
]


def team_http(calls=None, awards=AWARDS):
    routes = {
        "/team/frc2491": {"nickname": "NoMythic", "city": "Minneapolis", "state_prov": "Minnesota", "rookie_year": 2008},
        "/team/frc2491/events/simple": EV,
        "/team/frc2491/years_participated": [2008, 2019, 2025, 2026],
        "/team/frc2491/awards": awards,
        "/team/frc2491/events/2026/statuses": ST_2026,
        "/team/frc2491/events/2019/statuses": ST_2019,
        "/team/frc2491/events/2025/statuses": {"2025mndu": status(2, 55, 10, 2, 0, 1, 0, "f", "eliminated"), "2025mndu2": None},
        "/team/frc2491/awards/2019": [a for a in awards if a["year"] == 2019],
        "/team/frc2491/awards/2026": [a for a in awards if a["year"] == 2026],
        "/team/frc2491/event/2026mndu/status": ST_2026["2026mndu"],
        "/team/frc2491/event/2026mndu/awards": [awards[0]],
        "/team/frc2491/event/2026hop/status": ST_2026["2026hop"],
        "/team/frc2491/event/2026hop/awards": [],
        "/team/frc2491/event/2026cmptx/status": status(None, None, alliance=3, pick=1, level="f", pst="won"),
        "/team/frc2491/event/2026cmptx/awards": [awards[4]],
        "/team/frc2491/event/2025mndu/status": {"qual": {"num_teams": 55, "status": "completed",
                                                          "ranking": {"rank": 2, "record": {"wins": 10, "losses": 2, "ties": 0}}}},
        "/team/frc2491/event/2025mndu/awards": [],
        "/team/frc2491/event/2025mndu2/status": status(15, 55, 7, 5, 0),
        "/team/frc2491/event/2025mndu2/awards": [],
        "/team/frc2491/event/2019mndu/status": ST_2019["2019mndu"],
        "/team/frc2491/event/2019mndu/awards": [],
        "/team/frc2491/event/2019arc/status": ST_2019["2019arc"],
        "/team/frc2491/event/2019arc/awards": [],
    }

    def cond(url, headers):
        if calls is not None:
            calls.append(url)
        for k, v in routes.items():
            if url.endswith("/api/v3" + k):
                return v, None
        raise AssertionError("unexpected " + url)
    return cond


def tinfo(**args):
    f = FRC(cfg(frc_team="", frc_event=""), http_get=lambda *a, **k: None, http_get_cond=team_http())
    return f.handle("frc_team_info", {"team": "2491", **args})


def assert_clean(r, limit):
    s = r["say"]
    assert len(s.split()) <= limit, s
    assert "http" not in s and "2026mndu" not in s and "X-TBA" not in s
    assert "error" not in r


def test_event_by_city_duluth_prefers_requested_year_else_latest():
    r = tinfo(event="Duluth competition", year=2026)
    assert r["say"] == ("At the 2026 Lake Superior Regional in Duluth, team 2491 ranked 4th of 60 with a record of 9-3-0, "
                        "was the first pick of alliance 2 and lost in the semifinals. They won the Quality Award.")
    assert r["qual_rank"] == 4 and r["alliance"] == 2
    assert_clean(r, 60)
    print("SAY event:", r["say"])
    assert tinfo(event="their Duluth competition")["year"] == 2026          # most recent Duluth event
    assert "2019" in tinfo(event="Duluth", year=2019)["say"]


def test_event_ambiguous_same_year_names_the_other_one():
    r = tinfo(event="Duluth", year=2025)
    assert r["year"] == 2025 and r["also_matched"] == ["Lake Superior Regional"]
    assert "also played the Lake Superior Regional that year" in r["say"]
    assert "Northern Lights Regional" in r["say"] and "ranked 15th of 55 with a record of 7-5-0" in r["say"]
    assert "did not make the playoffs" in r["say"]
    print("SAY ambiguous:", r["say"])


def test_event_champs_and_houston_find_the_championship():
    for q in ("Champs in Houston", "Robotics Championships", "worlds", "championship", "Houston"):
        r = tinfo(event=q, year=2026)
        assert r["event"] == "Einstein Field", q                              # Einstein beats the division
        assert "won the event" in r["say"] and "They were the Championship Winners" in r["say"]
    r = tinfo(event="champs")
    assert r["year"] == 2026 and "also played the Hopper Division" in r["say"]
    r = tinfo(event="Hopper")
    assert r["event"] == "Hopper Division" and "ranked 30th of 75" in r["say"] and "did not make the playoffs" in r["say"]
    assert tinfo(event="Champs", year=2019)["event"] == "Archimedes Division"
    print("SAY champs:", tinfo(event="Champs in Houston")["say"])


def test_event_by_key_and_unknown_and_year_hint_in_text():
    assert tinfo(event="2026mndu")["event"] == "Lake Superior Regional"
    assert tinfo(event="duluth 2019")["year"] == 2019
    r = tinfo(event="Atlantis")
    assert "could not find an event matching Atlantis" in r["say"] and "Einstein Field in 2026" in r["say"]
    assert "error" not in r


def test_season_summary_for_a_past_year():
    r = tinfo(year=2019)
    assert r["say"].startswith("In 2019, team 2491 played 3 events.")
    assert_clean(r, 60)
    assert len(r["events"]) == 3 and r["events"][0]["rank"] == 9
    print("SAY season 2019:", r["say"])
    r = tinfo(year=2026)
    assert "won the event" in r["say"] or "Best was" in r["say"]
    assert_clean(r, 60)
    print("SAY season 2026:", r["say"])
    assert tinfo(year=1999)["say"] == "I have no events on record for team 2491 in 1999."


def test_career_summary_all():
    r = tinfo(detail="all")
    assert r["seasons"] == 4 and r["events"] == len(EV) and r["event_wins"] == 2 and r["finalist"] == 1
    assert r["championship_wins"] == [2026] and r["awards"]["Engineering Inspiration"] == 1
    assert [c["year"] for c in r["championships"]] == [2026, 2019]
    s = r["say"]
    assert s.index("won the Championship") < s.index("won 2 events") < s.index("went to the Championship 2 times")
    assert_clean(r, 70)
    print("SAY career:", s)
    assert tinfo(detail="everything")["seasons"] == 4


def test_default_summary_covers_all_latest_season_events():
    r = tinfo()
    assert "In 2026 they played 4 events." in r["say"]
    assert "Lake Superior Regional" in r["say"] or "Best was" in r["say"]
    assert r["name"] == "NoMythic" and "results_2026" in r and "recent_awards" in r
    assert_clean(r, 60)
    print("SAY default:", r["say"])


def test_team_lookup_say_lengths_no_secrets_and_cached():
    calls = []
    f = FRC(cfg(frc_team="", frc_event="", tba_api_key="SECRET-TBA-KEY"), http_get=lambda *a, **k: None,
            http_get_cond=team_http(calls))
    battery = ({}, {"year": 2019}, {"event": "Duluth"}, {"detail": "all"}, {"event": "champs"})
    for args in battery:
        r = f.handle("frc_team_info", {"team": "2491", **args})
        assert_clean(r, 70)
        assert "SECRET" not in str(r)
    # every lookup went through _tba -> cache: repeating the whole battery makes no new HTTP calls
    n = len(calls)
    for args in battery:
        f.handle("frc_team_info", {"team": "2491", **args})
    assert len(calls) == n and len(f._cache) > 5
    assert not any("nexus" in c for c in calls)


def test_event_and_year_args_via_python_api_and_bad_input():
    f = FRC(cfg(frc_team="2491"), http_get=lambda *a, **k: None, http_get_cond=team_http())
    assert f.team_info("frc2491", "2019")["year"] == 2019            # year as string, "frc" prefix
    assert f.team_info("", event="Duluth")["team"] == "2491"           # default team
    assert f.team_info("abc")["error"]


def test_lookalike_hosts_and_odd_team_text_are_refused():
    assert qr.parse_frc("https://evilfrc.nexus/2026mokc/254?key=x")["source"] == ""
    assert qr.parse_frc("https://www.frc.nexus/2026mokc/254?key=x")["event"] == "2026mokc"
    assert qr.parse_frc("https://notthebluealliance.com/team/254")["source"] == ""
    now = time.time() * 1000
    http = fake_http({"/event/2026mokc": nexus_event(now)})
    f = FRC(cfg(frc_team="1234", frc_event="2026mokc"), http_get=http)
    f.next_match("1/../../x?y")
    assert all("/../" not in u and "?" not in u for u, _ in http.calls)
