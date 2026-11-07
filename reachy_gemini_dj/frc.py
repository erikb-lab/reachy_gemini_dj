"""FRC match info for the robot: FRC Nexus (live queuing) and The Blue Alliance (rankings, results).

Data from Nexus: https://frc.nexus  (attribution requested by Nexus).  Data from The Blue Alliance:
https://www.thebluealliance.com. Keys stay on the robot; nothing here talks to Google.
"""
from __future__ import annotations

import json
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Callable

from . import qr

log = logging.getLogger(__name__)

NEXUS = "https://frc.nexus/api/v1"
TBA = "https://www.thebluealliance.com/api/v3"
ALERT_STATUSES = ("Queuing soon", "Now queuing", "On deck", "On field")
POLL_S = 30.0


def _get(url: str, headers: dict, timeout: float = 8.0) -> Any:
    req = urllib.request.Request(url, headers={"User-Agent": "reachy-gemini-dj", **headers})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _when(ms: float | None, now: float | None = None) -> str:
    if not ms:
        return ""
    mins = round((ms / 1000 - (now or time.time())) / 60)
    if mins > 1:
        return f"in about {mins} minutes"
    if mins >= -1:
        return "now"
    return f"{-mins} minutes ago"


class FRC:
    def __init__(self, config, get_frame: Callable[[], object] | None = None, say: Callable[[str], None] | None = None,
                 on_event: Callable[[str, str], None] | None = None, http_get=_get):
        self.config = config
        self.get_frame = get_frame
        self.say = say or (lambda t: None)
        self.on_event = on_event or (lambda k, t: None)
        self._get = http_get
        self.status: dict = {"now_queuing": "", "next_match": "", "error": "", "updated": 0}
        self._announced: dict[str, str] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ---------------- settings helpers ----------------
    @property
    def team(self) -> str:
        return str(getattr(self.config, "frc_team", "") or "").strip()

    @property
    def event(self) -> str:
        return str(getattr(self.config, "frc_event", "") or "").strip().lower()

    def _nexus(self, path: str) -> Any:
        key = (getattr(self.config, "nexus_api_key", "") or "").strip()
        if not key:
            raise RuntimeError("No Nexus API key set. Add one in Settings, FRC (get it at frc.nexus/api).")
        return self._get(NEXUS + path, {"Nexus-Api-Key": key})

    def _tba(self, path: str) -> Any:
        key = (getattr(self.config, "tba_api_key", "") or "").strip()
        if not key:
            raise RuntimeError("No Blue Alliance API key set. Add one in Settings, FRC (thebluealliance.com/account).")
        return self._get(TBA + path, {"X-TBA-Auth-Key": key})

    def _need_event(self) -> str:
        if not self.event:
            raise RuntimeError("No FRC event set. Scan your Nexus QR code or set the event code in Settings.")
        return self.event

    # ---------------- tools ----------------
    def handle(self, name: str, args: dict) -> dict:
        try:
            if name == "scan_qr":
                return self.scan_qr()
            if name == "frc_next_match":
                return self.next_match(str(args.get("team") or self.team))
            if name == "frc_event_status":
                return self.event_status()
            if name == "frc_team_info":
                return self.team_info(str(args.get("team") or self.team), args.get("year"))
            if name == "frc_rankings":
                return self.rankings(int(args.get("top") or 5))
            return {"error": f"Unknown FRC tool {name}"}
        except urllib.error.HTTPError as e:
            msg = {401: "the API key is missing", 403: "the API key was rejected", 404: "that event or team was not found"}
            return {"error": f"The FRC data service said no: {msg.get(e.code, f'HTTP {e.code}')}."}
        except (urllib.error.URLError, TimeoutError) as e:
            return {"error": f"Could not reach the FRC data service ({getattr(e, 'reason', e)})."}
        except Exception as e:
            return {"error": str(e)}

    def scan_qr(self) -> dict:
        if not self.get_frame:
            return {"error": "The camera is not available."}
        try:
            text = qr.scan(self.get_frame)
        except qr.MultipleCodes:
            return {"error": "I can see more than one QR code. Show me just your team's code."}
        if not text:
            return {"error": "I could not find a QR code. Hold it closer to my camera, flat and well lit."}
        info = qr.parse_frc(text)
        if not info["source"]:
            return {"qr_text": text[:200]}
        changed = []
        if info["source"] == "nexus":
            self.config.nexus_link = text          # private team link: kept on the robot, never spoken or sent
        if info["event"] and info["event"] != self.event:
            self.config.frc_event = info["event"]
            changed.append(f"event {info['event']}")
        if info["team"] and info["team"] != self.team:
            self.config.frc_team = info["team"]
            changed.append(f"team {info['team']}")
        try:
            self.config.save()
        except Exception as e:
            log.warning("could not save config: %s", e)
        self._announced.clear()
        self.on_event("frc", "Scanned a " + ("Nexus" if info["source"] == "nexus" else "Blue Alliance") + " code")
        if not (info["event"] or info["team"]):
            return {"result": "Saved the Nexus code, but it did not include an event or team. Set them in Settings."}
        return {"result": "Saved " + (" and ".join(changed) if changed else "the code (no changes)"),
                "team": self.team, "event": self.event}

    def _team_matches(self, data: dict, team: str) -> list[dict]:
        return [m for m in data.get("matches") or [] if team in (m.get("redTeams") or []) + (m.get("blueTeams") or [])]

    @staticmethod
    def _pending(m: dict, now: float | None = None) -> bool:
        """Not played yet. Nexus only records actual start times with AutoQueue, so a match whose estimated start
        is more than 10 minutes past (and that is not being queued) counts as done."""
        t = m.get("times") or {}
        if t.get("actualStartTime"):
            return False
        if m.get("status") in ("Queuing soon", "Now queuing", "On deck"):
            return True
        est = t.get("estimatedStartTime")
        return not (est and est / 1000 < (now or time.time()) - 600)

    def next_match(self, team: str) -> dict:
        ev = self._need_event()
        if not team:
            return {"error": "No team number set. Scan your Nexus QR code or set it in Settings."}
        try:
            data = self._nexus(f"/event/{ev}")
            upcoming = [m for m in self._team_matches(data, team) if self._pending(m)]
            if upcoming:
                m = upcoming[0]
                t = m.get("times") or {}
                side = "red" if team in (m.get("redTeams") or []) else "blue"
                return {"match": m.get("label"), "status": m.get("status") or "scheduled", "alliance": side,
                        "partners": [x for x in m.get(side + "Teams") or [] if x != team],
                        "opponents": m.get(("blue" if side == "red" else "red") + "Teams") or [],
                        "queue": _when(t.get("estimatedQueueTime")), "start": _when(t.get("estimatedStartTime")),
                        "now_queuing": data.get("nowQueuing") or ""}
            return {"result": f"Team {team} has no more scheduled matches on Nexus.",
                    "now_queuing": data.get("nowQueuing") or ""}
        except RuntimeError as e:
            if "Nexus API key" not in str(e):
                raise
        # No Nexus key: fall back to The Blue Alliance schedule
        ms = self._tba(f"/team/frc{team}/event/{ev}/matches/simple")
        order = {"qm": 0, "ef": 1, "qf": 2, "sf": 3, "f": 4}
        ms = sorted(ms, key=lambda m: (order.get(m.get("comp_level"), 9), m.get("set_number", 0), m.get("match_number", 0)))
        for m in ms:
            if m.get("actual_time") or (m.get("alliances", {}).get("red", {}).get("score", -1) not in (-1, None)):
                continue
            side = "red" if f"frc{team}" in m["alliances"]["red"]["team_keys"] else "blue"
            return {"match": f"{m['comp_level'].upper()} {m['match_number']}", "alliance": side,
                    "partners": [k[3:] for k in m["alliances"][side]["team_keys"] if k != f"frc{team}"],
                    "start": _when((m.get("predicted_time") or m.get("time") or 0) * 1000)}
        return {"result": f"Team {team} has no more scheduled matches."}

    def event_status(self) -> dict:
        ev = self._need_event()
        data = self._nexus(f"/event/{ev}")
        anns = [a.get("announcement", "") if isinstance(a, dict) else str(a) for a in data.get("announcements") or []]
        return {"event": ev, "now_queuing": data.get("nowQueuing") or "nothing yet",
                "announcements": [a[:160] for a in anns[-3:]]}

    def team_info(self, team: str, year: int | None = None) -> dict:
        """About a team: name, home, rookie year, recent events with results, awards. Needs only a Blue Alliance
        key; if an event is set (and a Nexus key) the pit location and current rank at it are added too."""
        team = (team or "").strip().lstrip("frcFRC") or self.team
        if not team.isdigit():
            return {"error": "Which team number?"}
        out: dict = {"team": team}
        try:
            t = self._tba(f"/team/frc{team}")
            out["name"] = t.get("nickname", "")
            out["from"] = ", ".join(x for x in (t.get("city"), t.get("state_prov")) if x)
            if t.get("rookie_year"):
                out["rookie_year"] = t["rookie_year"]
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            out["note"] = str(e)[:120]
        this_year = int(year or time.localtime().tm_year)
        try:
            events = self._tba(f"/team/frc{team}/events/simple") or []
            events = sorted(events, key=lambda e: e.get("start_date") or "", reverse=True)
            out["recent_events"] = [f"{e.get('name', '?')} ({e.get('year', '')})" for e in events[:6]]
            yr = this_year if any(e.get("year") == this_year for e in events) else (events[0].get("year") if events else 0)
            if yr:
                names = {e.get("key"): e.get("name", "") for e in events}
                statuses = self._tba(f"/team/frc{team}/events/{yr}/statuses") or {}
                results = []
                for key, st in statuses.items():
                    st = st or {}
                    q = ((st.get("qual") or {}).get("ranking") or {})
                    rec = q.get("record") or {}
                    row = {"event": names.get(key, key), "year": yr}
                    if q.get("rank"):
                        row["qual_rank"] = q["rank"]
                        row["record"] = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}-{rec.get('ties', 0)}"
                    po = (st.get("playoff") or {}).get("level")
                    if po:
                        row["playoffs"] = po
                    results.append(row)
                if results:
                    out["results_" + str(yr)] = results[:4]
        except urllib.error.HTTPError:
            raise
        except Exception as e:
            out.setdefault("note", str(e)[:120])
        try:
            aw = sorted(self._tba(f"/team/frc{team}/awards") or [], key=lambda a: a.get("year", 0), reverse=True)
            out["recent_awards"] = [f"{a.get('name', '')} ({a.get('year', '')})" for a in aw[:4]]
        except Exception:
            pass
        ev = self.event
        if ev:                                 # only when an event is set: where they are and how they are doing now
            if getattr(self.config, "nexus_api_key", ""):
                try:
                    out["pit"] = (self._nexus(f"/event/{ev}/pits") or {}).get(team) or "unknown"
                except Exception:
                    pass
            try:
                r = self._tba(f"/team/frc{team}/event/{ev}/status") or {}
                q = (r.get("qual") or {}).get("ranking") or {}
                if q.get("rank"):
                    rec = q.get("record") or {}
                    out["rank_at_event"] = q["rank"]
                    out["record_at_event"] = f"{rec.get('wins', 0)}-{rec.get('losses', 0)}-{rec.get('ties', 0)}"
            except Exception:
                pass
        return out

    def rankings(self, top: int = 5) -> dict:
        ev = self._need_event()
        rows = (self._tba(f"/event/{ev}/rankings") or {}).get("rankings") or []
        top = max(1, min(top, 10))
        res = [{"rank": r.get("rank"), "team": str(r.get("team_key", ""))[3:]} for r in rows[:top]]
        mine = next((r.get("rank") for r in rows if str(r.get("team_key", ""))[3:] == self.team), None)
        return {"top": res, **({"our_rank": mine} if mine else {})}

    # ---------------- spoken queuing alerts ----------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="frc-alerts")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def poll_once(self) -> None:
        if not (self.event and self.team and getattr(self.config, "nexus_api_key", "")):
            return
        data = self._nexus(f"/event/{self.event}")
        self.status.update(now_queuing=data.get("nowQueuing") or "", error="", updated=time.time())
        mine = [m for m in self._team_matches(data, self.team) if self._pending(m)]
        self.status["next_match"] = f"{mine[0].get('label')} ({mine[0].get('status') or 'scheduled'})" if mine else ""
        if not getattr(self.config, "frc_alerts", True):
            return
        for m in mine:
            label, st = m.get("label") or "", m.get("status") or ""
            if st in ALERT_STATUSES and self._announced.get(label) != st:
                first_seen = label not in self._announced
                self._announced[label] = st
                if st == "On field" and first_seen:
                    continue                    # do not shout about a match that is already being played
                side = "red" if self.team in (m.get("redTeams") or []) else "blue"
                phrase = {"Queuing soon": "is queuing soon", "Now queuing": "is now queuing",
                          "On deck": "is on deck", "On field": "is on the field"}[st]
                msg = f"Heads up! {label} {phrase}. We are on the {side} alliance."
                self.on_event("frc", msg)
                self.say(msg)

    def _loop(self) -> None:
        while not self._stop.wait(POLL_S if self.status.get("updated") else 3.0):
            try:
                self.poll_once()
            except Exception as e:
                self.status["error"] = str(e)[:160]
                self.status["updated"] = time.time()
