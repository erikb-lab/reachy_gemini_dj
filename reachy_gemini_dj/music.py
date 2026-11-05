"""Music streaming and playback subsystem.

Extracts stream URLs via yt-dlp and plays audio asynchronously using ffplay / ffmpeg.
Includes track queueing, volume control, and ducking for smooth voice interaction.
"""
from __future__ import annotations

import collections
import logging
import os
import signal
import subprocess
import threading
import time
from typing import Any

log = logging.getLogger(__name__)


class MusicPlayer:
    def __init__(self, volume: float = 0.70):
        self._volume = max(0.0, min(1.0, float(volume)))
        self._duck_factor = 1.0
        self._state = "idle"  # 'idle', 'playing', 'paused', 'loading'
        self._current_track = ""
        self._current_artist = ""
        self._duration = 0.0
        self._start_time = 0.0
        self._elapsed_paused = 0.0
        self._error = ""
        self._queue: collections.deque = collections.deque()
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._queue_worker, daemon=True)
        self._worker_thread.start()

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
            return {
                "state": self._state,
                "track": self._current_track,
                "artist": self._current_artist,
                "duration": self._duration,
                "elapsed": elapsed,
                "queued": len(self._queue),
                "error": self._error,
            }

    def set_volume(self, volume: float) -> None:
        with self._lock:
            self._volume = max(0.0, min(1.0, float(volume)))

    def set_duck(self, duck_level: float) -> None:
        with self._lock:
            self._duck_factor = max(0.0, min(1.0, float(duck_level)))

    def reset_duck(self) -> None:
        with self._lock:
            self._duck_factor = 1.0

    def play(self, query: str) -> dict[str, Any]:
        if not query.strip():
            return {"error": "No music query provided"}
        with self._lock:
            self._queue.append(query.strip())
            queued_count = len(self._queue)
            if self._state == "idle":
                self._state = "loading"
        return {"result": f"Queued: {query}", "queue_length": queued_count}

    def pause(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "playing" and self._proc:
                try:
                    self._proc.send_signal(signal.SIGSTOP)
                    self._elapsed_paused += time.time() - self._start_time
                    self._state = "paused"
                except Exception as e:
                    self._error = str(e)
            return {"state": self._state}

    def resume(self) -> dict[str, Any]:
        with self._lock:
            if self._state == "paused" and self._proc:
                try:
                    self._proc.send_signal(signal.SIGCONT)
                    self._start_time = time.time()
                    self._state = "playing"
                except Exception as e:
                    self._error = str(e)
            return {"state": self._state}

    def skip(self) -> dict[str, Any]:
        with self._lock:
            self._terminate_proc()
            self._state = "idle"
        return {"result": "Track skipped"}

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._queue.clear()
            self._terminate_proc()
            self._state = "idle"
            self._current_track = ""
            self._current_artist = ""
            self._duration = 0.0
            self._elapsed_paused = 0.0
        return {"result": "Music stopped"}

    def _terminate_proc(self) -> None:
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=1.0)
            except Exception:
                try:
                    self._proc.kill()
                except Exception:
                    pass
            self._proc = None

    def _queue_worker(self) -> None:
        while True:
            query = None
            with self._lock:
                if self._state in ("idle", "loading") and self._queue:
                    query = self._queue.popleft()
                    self._state = "loading"

            if not query:
                time.sleep(0.4)
                continue

            try:
                self._play_query(query)
            except Exception as e:
                log.exception("Playback failed for query: %s", query)
                with self._lock:
                    self._error = f"{type(e).__name__}: {e}"
                    self._state = "idle"

    def _play_query(self, query: str) -> None:
        import yt_dlp

        ydl_opts = {
            "format": "bestaudio/best",
            "noplaylist": True,
            "quiet": True,
            "default_search": "ytsearch",
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(query, download=False)
            if "entries" in info:
                info = info["entries"][0]
            stream_url = info.get("url")
            title = info.get("title", query)
            artist = info.get("uploader", "")
            duration = float(info.get("duration", 0))

        with self._lock:
            self._current_track = title
            self._current_artist = artist
            self._duration = duration
            self._start_time = time.time()
            self._elapsed_paused = 0.0
            self._state = "playing"

            cmd = [
                "ffplay",
                "-nodisp",
                "-autoexit",
                "-loglevel",
                "quiet",
                "-volume",
                str(int(self._volume * self._duck_factor * 100)),
                stream_url,
            ]
            self._proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if self._proc:
            self._proc.wait()

        with self._lock:
            self._proc = None
            if self._state == "playing":
                self._state = "idle"
