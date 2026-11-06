"""Pick and switch the voice backend: the local brain (free, default) or Gemini Live (cloud fallback).

config.backend:
  "local"  - only the local brain
  "gemini" - only Gemini Live
  "auto"   - local brain first; switch to Gemini Live if the brain is unreachable at start or fails
             FAIL_LIMIT turns/connects in a row (needs an API key), and switch back when the brain is
             reachable again and the robot is idle.
"""
from __future__ import annotations

import logging
import socket
import threading
import time
from urllib.parse import urlparse

log = logging.getLogger(__name__)

FAIL_LIMIT = 3
START_GRACE_S = 30.0
RETRY_LOCAL_EVERY_S = 300.0


def brain_reachable(url: str, timeout: float = 1.5) -> bool:
    u = urlparse(url or "ws://127.0.0.1:8765")
    try:
        with socket.create_connection((u.hostname or "127.0.0.1", u.port or 8765), timeout=timeout):
            return True
    except OSError:
        return False


class VoiceBackend:
    """Same surface main.py uses on GeminiSession, routed to whichever session is active."""

    def __init__(self, config, local, gemini, on_event, is_idle=lambda: True):
        self.config = config
        self.local = local
        self.gemini = gemini
        self.on_event = on_event
        self.is_idle = is_idle
        self.active = None
        self.reason = ""
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._started_at = 0.0
        self._last_local_try = 0.0
        self._watch: threading.Thread | None = None

    # ----- surface used by main.py -----
    @property
    def state(self) -> str:
        return self.active.state if self.active else "starting"

    @property
    def last_error(self) -> str:
        err = self.active.last_error if self.active else ""
        if not err and self.reason.endswith("(no Gemini key for fallback)"):
            err = self.reason                   # a real problem the user can fix, not just why we switched
        return err

    @property
    def active_name(self) -> str:
        return "local" if self.active is self.local else "gemini" if self.active is self.gemini else "none"

    @property
    def motion(self):
        return self.gemini.motion

    @motion.setter
    def motion(self, m) -> None:
        self.gemini.motion = m            # execute_tool lives on the Gemini session and is shared by both

    def push_mic(self, pcm: bytes) -> None:
        a = self.active
        if a:
            a.push_mic(pcm)

    def say(self, text: str) -> None:
        a = self.active
        if a and hasattr(a, "say"):
            a.say(text)

    def execute_tool(self, name, args):
        return self.gemini.execute_tool(name, args)

    def start(self) -> None:
        self._stop.clear()
        self._activate(self._preferred(), "startup")
        self._watch = threading.Thread(target=self._watchdog, daemon=True, name="backend-watch")
        self._watch.start()

    def stop(self) -> None:
        self._stop.set()
        for s in (self.local, self.gemini):
            try:
                s.stop()
            except Exception:
                pass
        self.active = None

    def init_client(self) -> bool:
        """Settings changed or new conversation: re-evaluate the mode, then reconnect the active session."""
        want = self._preferred()
        if want is not self.active and self._mode() != "auto":
            self._activate(want, "settings changed")
        elif self.active:
            if self.active is self.local:
                self.local.new_conversation()
            self.active.init_client()
        return True

    def status(self) -> dict:
        return {"mode": self._mode(), "active": self.active_name, "reason": self.reason,
                "brain_url": getattr(self.config, "brain_url", ""), "connected": bool(getattr(self.local, "connected", False)),
                "info": getattr(self.local, "info", {}), "error": getattr(self.local, "last_error", "")}

    # ----- switching -----
    def _mode(self) -> str:
        m = (getattr(self.config, "backend", "auto") or "auto").lower()
        return m if m in ("auto", "local", "gemini") else "auto"

    def _preferred(self):
        return self.gemini if self._mode() == "gemini" else self.local

    def _activate(self, sess, why: str) -> None:
        with self._lock:
            if sess is self.active:
                return
            old = self.active
            if old is not None:
                try:
                    old.stop()
                except Exception as e:
                    log.warning("stopping %s failed: %s", old, e)
                if old is self.gemini:            # GeminiSession.stop does not wait; let its thread finish
                    t = getattr(self.gemini, "_worker_thread", None)
                    if t:
                        t.join(5)
            self.active = sess
            self.reason = why
            self._started_at = time.monotonic()
            if sess is self.local:
                self._last_local_try = time.monotonic()
                self.local.consecutive_failures = 0
            sess.start()
        name = self.active_name
        log.info("voice backend -> %s (%s)", name, why)
        self.on_event("backend", f"Voice backend: {name} ({why})")

    def _local_failing(self) -> str:
        loc = self.local
        if loc.turn_overdue:
            return "local turn took too long"
        if loc.consecutive_failures >= FAIL_LIMIT:
            return f"local brain failed {loc.consecutive_failures} times: {loc.last_error}"
        if not loc.connected and time.monotonic() - self._started_at > START_GRACE_S and not loc.last_connect_ok:
            return f"local brain unreachable at {getattr(self.config, 'brain_url', '')}"
        return ""

    def _watchdog(self) -> None:
        while not self._stop.wait(2.0):
            try:
                mode = self._mode()
                if mode != "auto":
                    want = self._preferred()
                    if want is not self.active:
                        self._activate(want, f"mode set to {mode}")
                    continue
                if self.active is self.local:
                    why = self._local_failing()
                    if why:
                        if (self.config.api_key or "").strip():
                            self._activate(self.gemini, "fallback: " + why)
                        else:
                            self.reason = why + " (no Gemini key for fallback)"
                elif self.active is self.gemini:
                    due = time.monotonic() - self._last_local_try > RETRY_LOCAL_EVERY_S
                    if due and self.is_idle():
                        self._last_local_try = time.monotonic()
                        if brain_reachable(getattr(self.config, "brain_url", "")):
                            self._activate(self.local, "local brain is back")
            except Exception:
                log.exception("backend watchdog error")
