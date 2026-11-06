"""Music streaming and playback subsystem.

Extracts stream URLs via yt-dlp and plays audio asynchronously using ffplay / ffmpeg.
Includes track queueing, volume control, and ducking for smooth voice interaction.
"""
from __future__ import annotations

import atexit
import collections
import logging
import os
import re
import signal
import subprocess
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


def _clean_err(e: BaseException) -> str:
    """Short, single-line, ANSI-free error text suitable for the LLM to read aloud."""
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()
    return (msg.splitlines()[0] if msg else type(e).__name__)[:200]


class MusicPlayer:
    def __init__(self, volume: float = 0.70):
        self._volume = max(0.0, min(1.0, float(volume)))
        self._duck_factor = 1.0
        self._state = "idle"  # 'idle', 'playing', 'paused', 'loading'
        self._current_track = ""
        self._current_artist = ""
        self._duration = 0.0
        self._is_live = False
        self._start_time = 0.0
        self._elapsed_paused = 0.0
        self._error = ""
        self._queue: collections.deque = collections.deque()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        # Bumped by skip/stop to invalidate an in-flight lookup or running process.
        self._gen = 0
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self._worker_thread.start()
        atexit.register(self.shutdown)

    @property
    def is_playing(self) -> bool:
        with self._lock:
            return self._state == "playing"

    def get_status(self) -> dict[str, Any]:
        with self._lock:
            elapsed = 0.0
            if self._state == "playing":
                elapsed = self._elapsed_paused + (time.time() - self._start_time)
            elif self._state == "paused":
                elapsed = self._elapsed_paused
            if self._duration:
                elapsed = min(elapsed, self._duration)
            return {
                "state": self._state,
                "track": self._current_track,
                "artist": self._current_artist,
                "duration": self._duration,
                "is_live": self._is_live,
                "elapsed": elapsed,
                "queued": len(self._queue),
                "error": self._error,
            }

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(3.0, float(volume)))

    def set_duck(self, duck_level: float) -> None:
        with self._lock:
            self._duck_factor = max(0.0, min(1.0, float(duck_level)))

    def reset_duck(self) -> None:
        with self._lock:
            self._duck_factor = 1.0

    def play(self, query: str) -> dict[str, Any]:
        if not query or not query.strip():
            return {"error": "No music query provided"}
        query = query.strip()
        with self._cond:
            self._queue.append(query)
            queued_count = len(self._queue)
            if self._state == "idle":
                self._state = "loading"
            self._cond.notify_all()
        return {"result": f"Queued: {query}", "queue_length": queued_count}

    def pause(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            if self._state == "playing" and proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGSTOP)
                    self._elapsed_paused += time.time() - self._start_time
                    self._state = "paused"
                except Exception as e:
                    self._error = str(e)
                    return {"error": f"Could not pause: {_clean_err(e)}", "state": self._state}
            elif self._state != "paused":
                return {"error": "Nothing is playing", "state": self._state}
            return {"state": self._state}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            proc = self._proc
            if self._state == "paused" and proc is not None and proc.poll() is None:
                try:
                    proc.send_signal(signal.SIGCONT)
                    self._start_time = time.time()
                    self._state = "playing"
                except Exception as e:
                    self._error = str(e)
                    return {"error": f"Could not resume: {_clean_err(e)}", "state": self._state}
            elif self._state != "playing":
                return {"error": "Nothing is paused", "state": self._state}
            return {"state": self._state}

    def skip(self) -> dict[str, Any]:
        with self._cond:
            if self._state == "idle" and self._proc is None and not self._queue:
                return {"error": "Nothing is playing"}
            # Invalidate any in-flight lookup / running process; the worker then moves on to the queue.
            self._gen += 1
            proc = self._detach_proc()
            self._state = "idle"
            self._clear_track()
            self._cond.notify_all()
        self._terminate_proc(proc)
        return {"result": "Track skipped", "queued": self.get_status()["queued"]}

    def stop(self) -> dict[str, Any]:
        with self._cond:
            self._gen += 1
            self._queue.clear()
            proc = self._detach_proc()
            self._state = "idle"
            self._clear_track()
            self._cond.notify_all()
        self._terminate_proc(proc)
        return {"result": "Music stopped"}

    def shutdown(self) -> None:
        """Stop playback so no ffplay is left behind (also registered with atexit)."""
        try:
            self.stop()
        except Exception:
            pass

    def _clear_track(self) -> None:
        """Called with the lock held."""
        self._current_track = ""
        self._current_artist = ""
        self._duration = 0.0
        self._is_live = False
        self._elapsed_paused = 0.0

    def _detach_proc(self) -> "subprocess.Popen | None":
        """Called with the lock held; the caller terminates the returned process after releasing it."""
        proc = self._proc
        self._proc = None
        return proc

    @staticmethod
    def _terminate_proc(proc: "subprocess.Popen | None") -> None:
        """Terminate ffplay. Never call with the lock held (may block up to ~2s)."""
        if proc is None or proc.poll() is not None:
            return
        try:
            # A SIGSTOPped process will not act on SIGTERM until it is continued.
            proc.send_signal(signal.SIGCONT)
        except Exception:
            pass
        try:
            proc.terminate()
            proc.wait(timeout=1.0)
        except Exception:
            try:
                proc.kill()
                proc.wait(timeout=1.0)
            except Exception:
                pass

    def _queue_worker(self) -> None:
        while True:
            with self._cond:
                while not (self._queue and self._state in ("idle", "loading")):
                    self._cond.wait(timeout=1.0)
                query = self._queue.popleft()
                self._state = "loading"
                gen = self._gen

            try:
                self._play_query(query, gen)
            except Exception as e:
                log.exception("Playback failed for query: %s", query)
                with self._lock:
                    if gen == self._gen:
                        self._error = f"Couldn't play '{query}': {_clean_err(e)}"
                        self._state = "idle"
                        self._clear_track()

    @staticmethod
    def _extract(query: str) -> dict[str, Any]:
        """Resolve a query to a stream URL with yt-dlp. Streaming only: nothing is downloaded."""
        import yt_dlp

        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "no_warnings": True,
            "default_search": "ytsearch",
            "socket_timeout": 15,
            "extractor_retries": 1,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
        if not info:
            raise RuntimeError("no results found")
        if info.get("entries") is not None:
            entry = next((e for e in info["entries"] if e), None)
            if entry is None:
                raise RuntimeError("no results found")
            info = entry
        stream_url = info.get("url")
        headers = info.get("http_headers")
        if not stream_url:
            for f in reversed(info.get("requested_formats") or info.get("formats") or []):
                if f.get("url") and f.get("acodec") != "none":
                    stream_url = f["url"]
                    headers = f.get("http_headers") or headers
                    break
        if not stream_url:
            raise RuntimeError("no playable stream URL found")
        try:
            duration = float(info.get("duration") or 0)  # None for livestreams
        except (TypeError, ValueError):
            duration = 0.0
        return {
            "url": stream_url,
            "headers": dict(headers or {}),
            "title": info.get("title") or query,
            "artist": info.get("uploader") or info.get("channel") or "",
            "duration": duration,
            "live": bool(info.get("is_live")),
        }

    def _play_query(self, query: str, gen: int) -> None:
        with self._lock:
            self._error = ""
        info = self._extract(query)
        stream_url = info["url"]
        headers = info["headers"]
        title = info["title"]
        artist = info["artist"]
        duration = info["duration"]

        with self._lock:
            if gen != self._gen:
                return  # skipped/stopped while the lookup was in flight
            self._current_track = title
            self._current_artist = artist
            self._duration = duration
            self._is_live = info["live"]
            self._start_time = time.time()
            self._elapsed_paused = 0.0
            self._state = "playing"

            effective_vol = self._volume * self._duck_factor
            header_str = "".join(f"{k}: {v}\r\n" for k, v in headers.items()) if headers else ""
            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "warning",
            ]
            if effective_vol > 1.0:
                cmd += ["-volume", "100", "-af", f"volume={effective_vol:.2f}"]
            else:
                cmd += ["-volume", str(int(effective_vol * 100))]

            if header_str:
                cmd += ["-headers", header_str]
            cmd += [
                "-reconnect",
                "1",
                "-reconnect_streamed",
                "1",
                "-reconnect_delay_max",
                "5",
                stream_url,
            ]
            log.info("Starting ffplay for %s (vol=%.2fx)", title, effective_vol)
            try:
                proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            except Exception as e:
                self._state = "idle"
                self._clear_track()
                self._error = f"Couldn't start ffplay: {_clean_err(e)}"
                return
            self._proc = proc

        err = ""
        try:
            _, err = proc.communicate()
        except Exception:
            pass
        if err and "error" in err.lower():
            log.warning("ffplay note: %s", err[:200])

        with self._lock:
            if self._proc is proc:
                self._proc = None
            if gen != self._gen:
                return  # skip/stop already updated state
            if proc.returncode not in (0, None):
                self._error = f"Playback of '{title}' failed: {(err or 'ffplay error').strip()[-200:]}"
            self._state = "idle"
            self._clear_track()
