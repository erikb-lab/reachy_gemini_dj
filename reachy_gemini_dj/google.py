"""Google account connection: Calendar, Tasks and Gmail (read-only) over plain HTTPS.

No Google client libraries (they are heavy for the CM4). Sign-in uses the "Desktop app" OAuth flow with
PKCE: the settings page opens Google's consent screen, Google then redirects to http://localhost (which
fails to load on your computer, that is expected) and you paste that address back into the page.
Tokens live in ~/.reachy_gemini_dj/google.json (mode 600), never in the code.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import secrets
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import HOME

log = logging.getLogger(__name__)

FILE = HOME / "google.json"
REDIRECT_URI = "http://localhost:8765/"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"

SERVICES = {
    "calendar": {
        "label": "Calendar",
        "scopes": [
            "https://www.googleapis.com/auth/calendar.events",
            "https://www.googleapis.com/auth/calendar.readonly",
        ],
    },
    "tasks": {"label": "Tasks", "scopes": ["https://www.googleapis.com/auth/tasks"]},
    "gmail": {"label": "Gmail (read only)", "scopes": ["https://www.googleapis.com/auth/gmail.readonly"]},
}
BASE_SCOPES = ["openid", "email"]


class GoogleError(Exception):
    pass


def _local_tz_dt(s: str) -> datetime:
    """Parse an ISO date-time; a time without a zone is taken as the robot's local time."""
    s = s.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(s)
    return dt.astimezone() if dt.tzinfo is None else dt


def _human_time(start: dict, end: dict) -> str:
    try:
        if "date" in start:
            d = date.fromisoformat(start["date"])
            return f"{d.strftime('%a %b %d')}, all day"
        a = datetime.fromisoformat(start["dateTime"].replace("Z", "+00:00")).astimezone()
        b = datetime.fromisoformat(end["dateTime"].replace("Z", "+00:00")).astimezone()
        fmt = "%I:%M %p".replace("%I", "%-I")
        return f"{a.strftime('%a %b %d')}, {a.strftime(fmt)} to {b.strftime(fmt)}"
    except Exception:
        return str(start.get("dateTime") or start.get("date") or "")


class GoogleAccount:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict = {}
        self._pending: dict | None = None
        self._load()

    # ------------------------------------------------------------ storage ---
    def _load(self) -> None:
        try:
            if FILE.exists():
                self._data = json.loads(FILE.read_text())
        except Exception as e:
            log.warning("Could not read google.json (%s)", e)
            self._data = {}

    def _save(self) -> None:
        HOME.mkdir(parents=True, exist_ok=True)
        tmp = FILE.with_suffix(".tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(self._data, f, indent=2)
        os.replace(tmp, FILE)

    # ------------------------------------------------------------- status ---
    @property
    def connected(self) -> bool:
        return bool(self._data.get("refresh_token"))

    def granted(self) -> list[str]:
        """Service names (calendar/tasks/gmail) whose scopes were all granted."""
        have = set((self._data.get("scope") or "").split())
        return [k for k, v in SERVICES.items() if all(s in have for s in v["scopes"])]

    def has(self, service: str) -> bool:
        return self.connected and service in self.granted()

    def public(self) -> dict:
        return {
            "client_set": bool(self._data.get("client_id") and self._data.get("client_secret")),
            "client_id_hint": (self._data.get("client_id") or "")[:12],
            "connected": self.connected,
            "email": self._data.get("email", ""),
            "services": self.granted(),
            "needs_reconnect": bool(self._data.get("needs_reconnect")),
            "available": {k: v["label"] for k, v in SERVICES.items()},
        }

    # ------------------------------------------------------------ sign-in ---
    def set_client(self, client_id: str, client_secret: str) -> None:
        cid, sec = client_id.strip(), client_secret.strip()
        if not cid or not sec:
            raise GoogleError("Enter both the client ID and the client secret.")
        with self._lock:
            self._data["client_id"], self._data["client_secret"] = cid, sec
            self._save()

    def begin(self, services: list[str]) -> str:
        if not (self._data.get("client_id") and self._data.get("client_secret")):
            raise GoogleError("Save your Google client ID and secret first.")
        services = [s for s in services if s in SERVICES] or ["calendar"]
        scopes = list(BASE_SCOPES)
        for s in services:
            scopes += SERVICES[s]["scopes"]
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        state = secrets.token_urlsafe(16)
        self._pending = {"verifier": verifier, "state": state}
        q = {
            "client_id": self._data["client_id"],
            "redirect_uri": REDIRECT_URI,
            "response_type": "code",
            "scope": " ".join(scopes),
            "access_type": "offline",
            "prompt": "consent",
            "include_granted_scopes": "true",
            "state": state,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
        return AUTH_URL + "?" + urllib.parse.urlencode(q)

    def finish(self, pasted: str) -> dict:
        pend = self._pending
        if not pend:
            raise GoogleError("Press Connect first, then paste the address you were sent to.")
        pasted = pasted.strip()
        code, state = pasted, None
        if "code=" in pasted or pasted.startswith("http"):
            qs = urllib.parse.parse_qs(urllib.parse.urlparse(pasted).query)
            if "error" in qs:
                raise GoogleError(f"Google said: {qs['error'][0]}")
            code = (qs.get("code") or [""])[0]
            state = (qs.get("state") or [None])[0]
        if not code:
            raise GoogleError("That address has no code in it. Copy the whole address from the browser bar.")
        if state and state != pend["state"]:
            raise GoogleError("That sign-in belongs to an older attempt. Press Connect again.")
        tok = self._post(
            TOKEN_URL,
            {
                "code": code,
                "client_id": self._data["client_id"],
                "client_secret": self._data["client_secret"],
                "redirect_uri": REDIRECT_URI,
                "grant_type": "authorization_code",
                "code_verifier": pend["verifier"],
            },
        )
        with self._lock:
            if tok.get("refresh_token"):
                self._data["refresh_token"] = tok["refresh_token"]
            self._data["access_token"] = tok["access_token"]
            self._data["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
            self._data["scope"] = tok.get("scope", "")
            self._data.pop("needs_reconnect", None)
            self._save()
        self._pending = None
        try:
            info = self._call("GET", USERINFO_URL)
            with self._lock:
                self._data["email"] = info.get("email", "")
                self._save()
        except GoogleError:
            pass
        return self.public()

    def disconnect(self) -> None:
        tok = self._data.get("refresh_token")
        if tok:
            try:
                self._post(REVOKE_URL, {"token": tok})
            except GoogleError:
                pass
        with self._lock:
            keep = {k: self._data[k] for k in ("client_id", "client_secret") if k in self._data}
            self._data = keep
            self._save()

    # ------------------------------------------------------------- http ---
    @staticmethod
    def _post(url: str, form: dict) -> dict:
        req = urllib.request.Request(
            url,
            data=urllib.parse.urlencode(form).encode(),
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode() or "{}"
                return json.loads(raw)
        except urllib.error.HTTPError as e:
            try:
                body = json.loads(e.read().decode())
                msg = body.get("error_description") or body.get("error") or str(e)
                if body.get("error") == "invalid_grant":
                    msg = "invalid_grant: " + msg
            except Exception:
                msg = str(e)
            raise GoogleError(msg) from None
        except Exception as e:
            raise GoogleError(f"Could not reach Google: {e}") from None

    def _access_token(self) -> str:
        with self._lock:
            if not self.connected:
                raise GoogleError("Google is not connected.")
            if self._data.get("access_token") and time.time() < self._data.get("expires_at", 0):
                return self._data["access_token"]
            form = {
                "client_id": self._data["client_id"],
                "client_secret": self._data["client_secret"],
                "refresh_token": self._data["refresh_token"],
                "grant_type": "refresh_token",
            }
        try:
            tok = self._post(TOKEN_URL, form)
        except GoogleError as e:
            if "invalid_grant" in str(e):
                with self._lock:
                    self._data["needs_reconnect"] = True
                    self._data.pop("access_token", None)
                    self._save()
                raise GoogleError("Google sign-in expired. Reconnect Google in the settings page.") from None
            raise
        with self._lock:
            self._data["access_token"] = tok["access_token"]
            self._data["expires_at"] = time.time() + int(tok.get("expires_in", 3600)) - 60
            self._save()
            return self._data["access_token"]

    def _call(self, method: str, url: str, params: dict | None = None, body: dict | None = None) -> dict:
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params, doseq=True)
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Authorization": f"Bearer {self._access_token()}", "Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        req = urllib.request.Request(url, data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=15) as r:
                raw = r.read().decode()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            try:
                err = json.loads(e.read().decode()).get("error", {})
                msg = err.get("message") if isinstance(err, dict) else str(err)
            except Exception:
                msg = str(e)
            raise GoogleError(f"Google API error {e.code}: {msg}"[:200]) from None
        except Exception as e:
            raise GoogleError(f"Could not reach Google: {e}") from None

    def _need(self, service: str) -> None:
        if not self.connected:
            raise GoogleError("Google is not connected.")
        if service not in self.granted():
            raise GoogleError(f"{SERVICES[service]['label']} access was not granted. Reconnect Google and tick it.")

    # ---------------------------------------------------------- calendar ---
    def calendar_events(self, days: float = 1, max_results: int = 8, start_offset_days: float = 0) -> dict:
        self._need("calendar")
        now = datetime.now().astimezone()
        t0 = now + timedelta(days=start_offset_days)
        if start_offset_days >= 1:
            t0 = t0.replace(hour=0, minute=0, second=0, microsecond=0)
        t1 = t0 + timedelta(days=max(0.1, min(float(days), 31)))
        res = self._call(
            "GET",
            "https://www.googleapis.com/calendar/v3/calendars/primary/events",
            {
                "timeMin": t0.isoformat(),
                "timeMax": t1.isoformat(),
                "singleEvents": "true",
                "orderBy": "startTime",
                "maxResults": max(1, min(int(max_results), 20)),
            },
        )
        events = []
        for e in res.get("items", []):
            if e.get("status") == "cancelled":
                continue
            item = {"title": e.get("summary") or "(no title)", "when": _human_time(e.get("start", {}), e.get("end", {}))}
            if e.get("location"):
                item["where"] = e["location"][:80]
            events.append(item)
        return {"events": events, "count": len(events)}

    def calendar_add(
        self,
        title: str,
        start: str,
        duration_minutes: int = 60,
        all_day: bool = False,
        location: str = "",
        description: str = "",
    ) -> dict:
        self._need("calendar")
        if not title.strip():
            raise GoogleError("The event needs a title.")
        try:
            a = _local_tz_dt(start)
        except ValueError:
            raise GoogleError("Could not understand the start time. Use an ISO date and time.") from None
        body: dict = {"summary": title.strip()[:200]}
        if all_day:
            d = a.date()
            body["start"], body["end"] = {"date": d.isoformat()}, {"date": (d + timedelta(days=1)).isoformat()}
        else:
            b = a + timedelta(minutes=max(5, min(int(duration_minutes or 60), 24 * 60)))
            body["start"], body["end"] = {"dateTime": a.isoformat()}, {"dateTime": b.isoformat()}
        if location:
            body["location"] = location[:200]
        if description:
            body["description"] = description[:1000]
        ev = self._call("POST", "https://www.googleapis.com/calendar/v3/calendars/primary/events", body=body)
        return {
            "result": "Added to your calendar",
            "title": ev.get("summary", title),
            "when": _human_time(ev.get("start", {}), ev.get("end", {})),
        }

    # ------------------------------------------------------------- tasks ---
    def tasks_list(self, max_results: int = 10) -> dict:
        self._need("tasks")
        res = self._call(
            "GET",
            "https://tasks.googleapis.com/tasks/v1/lists/@default/tasks",
            {"showCompleted": "false", "maxResults": max(1, min(int(max_results), 30))},
        )
        out = []
        for t in res.get("items", []):
            item = {"title": t.get("title", "")}
            if t.get("due"):
                item["due"] = t["due"][:10]
            out.append(item)
        return {"tasks": out, "count": len(out)}

    def tasks_add(self, title: str, due: str = "") -> dict:
        self._need("tasks")
        if not title.strip():
            raise GoogleError("The task needs a title.")
        body = {"title": title.strip()[:200]}
        if due:
            try:
                d = date.fromisoformat(due.strip()[:10])
                body["due"] = f"{d.isoformat()}T00:00:00.000Z"
            except ValueError:
                pass
        t = self._call("POST", "https://tasks.googleapis.com/tasks/v1/lists/@default/tasks", body=body)
        return {"result": "Task added", "title": t.get("title", title)}

    def tasks_complete(self, title: str) -> dict:
        self._need("tasks")
        res = self._call(
            "GET",
            "https://tasks.googleapis.com/tasks/v1/lists/@default/tasks",
            {"showCompleted": "false", "maxResults": 100},
        )
        q = title.strip().lower()
        match = next((t for t in res.get("items", []) if q and q in (t.get("title", "").lower())), None)
        if not match:
            raise GoogleError("No open task matches that.")
        self._call(
            "PATCH",
            f"https://tasks.googleapis.com/tasks/v1/lists/@default/tasks/{match['id']}",
            body={"status": "completed"},
        )
        return {"result": "Marked done", "title": match.get("title", "")}

    # ------------------------------------------------------------- gmail ---
    def gmail(self, query: str = "is:unread in:inbox", max_results: int = 5) -> dict:
        self._need("gmail")
        n = max(1, min(int(max_results), 8))
        lst = self._call(
            "GET",
            "https://gmail.googleapis.com/gmail/v1/users/me/messages",
            {"q": query or "is:unread in:inbox", "maxResults": n},
        )
        out = []
        for m in lst.get("messages", [])[:n]:
            d = self._call(
                "GET",
                f"https://gmail.googleapis.com/gmail/v1/users/me/messages/{m['id']}",
                {"format": "metadata", "metadataHeaders": ["From", "Subject"]},
            )
            h = {x["name"].lower(): x["value"] for x in d.get("payload", {}).get("headers", [])}
            sender = h.get("from", "")
            sender = sender.split("<")[0].strip().strip('"') or sender
            out.append(
                {
                    "from": sender[:60],
                    "subject": h.get("subject", "(no subject)")[:100],
                    "preview": (d.get("snippet") or "")[:140],
                }
            )
        return {"emails": out, "count": len(out)}
