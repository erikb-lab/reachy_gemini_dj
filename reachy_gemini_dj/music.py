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

import itertools
import random

from . import favorites as _fav
from . import gain as _gain

log = logging.getLogger(__name__)

STREAM_TTL = 4 * 3600          # resolved stream URLs expire; re-resolve when older than this
MAX_TRACK_S = 15 * 60          # skip longer videos unless the query asks for a mix/playlist/live
RECENT_MAX = 50
VOLUME_STEP = 0.15             # adjust_volume: 15 percentage points
_LONG_OK = re.compile(r"\b(mix|playlist|live|album|hours?|full|set|radio|stream|compilation)\b", re.I)
_ids = itertools.count(1)


def speakable(res: dict[str, Any]) -> dict[str, Any]:
    """Errors carry the speakable sentence (the brain reads `error` aloud when there is one)."""
    if isinstance(res, dict) and res.get("error") and res.get("say"):
        res["error"] = res.pop("say")
    return res


def _clean_query(q: str) -> str:
    return re.sub(r"\s+(songs?|music|tracks?)$", "", q.strip(), flags=re.I) or q.strip()


def _clean_err(e: BaseException) -> str:
    """Short, single-line, ANSI-free error text suitable for the LLM to read aloud."""
    msg = re.sub(r"\x1b\[[0-9;]*m", "", str(e)).replace("ERROR: ", "").strip()
    return (msg.splitlines()[0] if msg else type(e).__name__)[:200]


class MusicPlayer:
    def __init__(self, volume: float = 0.70, autoplay: bool = False):
        # Linear multiplier, 0.0-2.5 (see gain.py for the scale); duck is 0.0-1.0.
        self._volume = _gain.clamp_volume(volume)
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
        self._current_item: dict[str, Any] | None = None
        self._current_info: dict[str, Any] | None = None
        self._seek_req: float | None = None
        # Autoplay: after the queue runs dry, keep going with results related to the last request.
        self.autoplay = bool(autoplay)
        self._auto_active = False       # a request is "live" (cleared by stop / clear)
        self._auto_query = ""
        self._auto_n = 10
        self._auto_done = False
        self._auto_pool: list[dict[str, Any]] = []
        self._recent: collections.deque = collections.deque(maxlen=RECENT_MAX)   # recently played video ids
        self._fail_streak = 0
        self._prefetching = False
        self._prefetch_again = False
        self._seek_paused = False
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
                "queue_items": [self._public_item(i) for i in self._queue],
                "autoplay": self.autoplay,
                "error": self._error,
            }

    def set_volume(self, volume: float) -> None:
        """Set music volume (linear multiplier 0.0-2.5); applied live, ramped smoothly."""
        with self._lock:
            self._volume = _gain.clamp_volume(volume)

    def set_duck(self, duck_level: float) -> None:
        """Duck music to this fraction (0.0-1.0) of volume; fades in over ~80 ms."""
        with self._lock:
            self._duck_factor = _gain.clamp_duck(duck_level)

    def _target_gain(self) -> float:
        """Total gain the audio pump should be heading towards (volume * duck)."""
        with self._lock:
            return self._volume * self._duck_factor

    def reset_duck(self) -> None:
        with self._lock:
            self._duck_factor = 1.0

    # ---------- queue items ----------
    @staticmethod
    def _new_item(query: str, added_by: str = "voice", entry: dict | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {"id": next(_ids), "query": query, "added_by": added_by}
        if entry:
            item.update(title=entry.get("title") or "", artist=entry.get("artist") or "",
                        video_id=entry.get("id") or "", url=entry.get("url") or "")
        return item

    @staticmethod
    def _public_item(i: dict[str, Any]) -> dict[str, Any]:
        title = i.get("title") or i.get("query") or ""
        if i.get("expand"):
            title = f"{title} (x{i['expand']})"
        return {"id": i["id"], "title": title, "artist": i.get("artist") or "", "added_by": i.get("added_by", "")}

    def _fresh(self, item: dict[str, Any]) -> bool:
        return bool(item.get("info")) and time.time() - item.get("info_ts", 0) < STREAM_TTL

    def _set_state_loading(self) -> None:
        """Called with the lock held."""
        if self._state == "idle":
            self._state = "loading"

    def play(self, query: str, count: int = 1, when: str = "", added_by: str = "voice") -> dict[str, Any]:
        """Queue music. count: 1-10 different tracks for the query. when: 'now' | 'next' | 'last' (default)."""
        if not query or not str(query).strip():
            return {"error": "No music query provided", "say": "What should I play?"}
        query = str(query).strip()
        try:
            count = max(1, min(10, int(count)))
        except (TypeError, ValueError):
            count = 1
        w = str(when or "").strip().lower()
        w = "now" if w in ("now", "immediately", "right now") else "next" if w in ("next", "front", "first") else "last"
        item = self._new_item(query, added_by)
        if count > 1:
            item["expand"] = count
            item["pending"] = True
        with self._cond:
            idle = self._state == "idle"
            self._auto_active, self._auto_query = True, query
            self._auto_n, self._auto_done, self._auto_pool = 10, False, []
            if w == "last":
                self._queue.append(item)
            else:
                self._queue.appendleft(item)
            queued_count = len(self._queue)
            self._set_state_loading()
            self._cond.notify_all()
        if w == "now" and not idle:
            self._interrupt()
        if count > 1:
            threading.Thread(target=self._expand, args=(item,), daemon=True, name="music-expand").start()
        self._kick_prefetch()
        q = _clean_query(query)
        if count > 1:
            verb = "playing" if (w == "now" or (idle and w == "last")) else "queuing"
            say = f"Okay, {verb} {count} {q} songs" + (" now." if w == "now" else " next." if w == "next" and verb == "queuing" else ".")
        elif w == "now":
            say = f"Playing {q} now."
        elif w == "next":
            say = "Okay, that's up next."
        else:
            say = f"Okay, playing {q}." if idle else "Okay, queuing that up."
        return {"result": f"Queued: {query}", "queue_length": queued_count, "say": say}

    def _interrupt(self) -> None:
        """Stop the current track/lookup without touching the queue (the worker moves on to the queue head)."""
        with self._cond:
            self._gen += 1
            proc = self._detach_proc()
            self._state = "idle"
            self._clear_track()
            self._cond.notify_all()
        self._terminate_proc(proc)

    @staticmethod
    def _filter_entries(entries: list[dict], query: str) -> list[dict]:
        """Drop duplicates and (unless the query asks for a mix/live) very long videos."""
        long_ok = bool(_LONG_OK.search(query))
        seen_ids: set[str] = set()
        seen_titles: set[str] = set()
        out = []
        for e in entries:
            vid, title = e.get("id") or "", (e.get("title") or "").strip().lower()
            if (vid and vid in seen_ids) or (title and title in seen_titles):
                continue
            dur = e.get("duration") or 0
            if not long_ok and (e.get("live") or dur > MAX_TRACK_S):
                continue
            seen_ids.add(vid)
            seen_titles.add(title)
            out.append(e)
        return out

    def _expand(self, placeholder: dict[str, Any]) -> None:
        """Replace a count>1 placeholder in the queue with that many different tracks (one yt-dlp search)."""
        count, query = placeholder["expand"], placeholder["query"]
        try:
            entries = self._filter_entries(self._search(query, min(30, count * 2 + 3)), query)
        except Exception as e:
            log.warning("Multi-track search failed for %r: %s", query, e)
            entries = []
        with self._cond:
            lst = list(self._queue)
            idx = next((n for n, i in enumerate(lst) if i["id"] == placeholder["id"]), None)
            if idx is None:
                return  # cleared / removed meanwhile
            taken = {i.get("video_id") for i in lst if i.get("video_id")}
            picks = [e for e in entries if not (e.get("id") and e["id"] in taken)][:count]
            new = [self._new_item(query, placeholder["added_by"], e) for e in picks]
            if not new:   # search failed: fall back to one plain query so something still plays
                new = [self._new_item(query, placeholder["added_by"])]
            lst[idx:idx + 1] = new
            self._queue = collections.deque(lst)
            self._cond.notify_all()
        self._kick_prefetch()

    def enqueue_entries(self, entries: list[dict], when: str = "last", added_by: str = "voice") -> int:
        """Queue already-known tracks (id/title/artist/url dicts), e.g. favorites. Clears the autoplay seed."""
        items = [self._new_item(e.get("title") or "", added_by, e) for e in entries]
        if not items:
            return 0
        with self._cond:
            self._auto_active, self._auto_query, self._auto_pool = True, "", []
            if when == "next":
                self._queue.extendleft(reversed(items))
            else:
                self._queue.extend(items)
            self._set_state_loading()
            self._cond.notify_all()
        self._kick_prefetch()
        return len(items)

    # ---------- queue control (music_queue tool) ----------
    def queue_action(self, action: str, which: str = "") -> dict[str, Any]:
        action = (action or "").strip().lower()
        which = str(which or "").strip().lower()
        if action == "now_playing":
            st = self.get_status()
            if st["state"] == "idle" or not st["track"]:
                return {"result": "Nothing is playing", "say": "Nothing is playing."}
            by = f" by {st['artist']}" if st["artist"] else ""
            return {"result": st["track"], "track": st["track"], "artist": st["artist"],
                    "say": f"This is {st['track']}{by}."}
        if action == "list":
            with self._lock:
                names = [self._public_item(i)["title"] for i in self._queue]
            n = len(names)
            if not n:
                return {"result": "Queue is empty", "queue": [],
                        "say": "The queue is empty." if not self.autoplay else "The queue is empty, but I'll keep the music going."}
            if n <= 3:
                say = "Up next: " + ", then ".join(names) + "."
            else:
                say = f"Up next: {names[0]}, then {names[1]}, and {n - 2} more."
            return {"result": f"{n} queued", "queue": names, "say": say}
        if action == "clear":
            with self._cond:
                n = len(self._queue)
                self._queue.clear()
                self._auto_active, self._auto_pool = False, []
            return {"result": f"Cleared {n}", "say": "Cleared the queue." if n else "The queue is already empty."}
        if action == "shuffle":
            with self._cond:
                lst = list(self._queue)
                if len(lst) < 2:
                    return {"result": "Nothing to shuffle", "say": "There's nothing to shuffle."}
                random.shuffle(lst)
                self._queue = collections.deque(lst)
            self._kick_prefetch()
            return {"result": "Shuffled", "say": "Shuffled the queue."}
        if action == "remove":
            with self._cond:
                lst = list(self._queue)
                idx = self._find(lst, which)
                if idx is None:
                    return {"error": "Not in the queue", "say": "I couldn't find that in the queue."}
                gone = lst.pop(idx)
                self._queue = collections.deque(lst)
            self._kick_prefetch()
            name = self._public_item(gone)["title"]
            return {"result": f"Removed {name}", "say": f"Removed {name}."}
        if action in ("autoplay_on", "autoplay_off"):
            self.autoplay = action == "autoplay_on"
            with self._cond:
                self._cond.notify_all()
            return {"result": "ok", "autoplay": self.autoplay,
                    "say": "Okay, I'll keep the music going." if self.autoplay else "Okay, I'll stop when the queue ends."}
        return {"error": f"Unknown action: {action}", "say": "I can't do that with the queue."}

    def remove_item(self, item_id: int) -> dict[str, Any]:
        with self._cond:
            lst = list(self._queue)
            idx = next((n for n, i in enumerate(lst) if i["id"] == item_id), None)
            if idx is None:
                return {"error": "Not in the queue", "say": "I couldn't find that in the queue."}
            lst.pop(idx)
            self._queue = collections.deque(lst)
        return {"result": "Removed"}

    def move_next(self, item_id: int) -> dict[str, Any]:
        with self._cond:
            lst = list(self._queue)
            idx = next((n for n, i in enumerate(lst) if i["id"] == item_id), None)
            if idx is None:
                return {"error": "Not in the queue", "say": "I couldn't find that in the queue."}
            lst.insert(0, lst.pop(idx))
            self._queue = collections.deque(lst)
            self._cond.notify_all()
        self._kick_prefetch()
        return {"result": "Moved to next"}

    @staticmethod
    def _find(lst: list[dict], which: str) -> int | None:
        if not lst or not which:
            return None
        if which in ("next", "first"):
            return 0
        if which in ("last", "end"):
            return len(lst) - 1
        if which.isdigit():
            n = int(which)
            return n - 1 if 1 <= n <= len(lst) else None
        words = re.findall(r"\w+", which)
        for n, i in enumerate(lst):
            hay = " ".join((i.get("title") or "", i.get("artist") or "", i.get("query") or "")).lower()
            if words and all(w in hay for w in words):
                return n
        return None

    def adjust_volume(self, direction: str) -> dict[str, Any]:
        """One 15-point step up/down on the percent scale. Caller persists config.music_volume from result['volume']."""
        d = (direction or "").strip().lower()
        if d not in ("up", "down"):
            return {"error": "direction must be up or down", "say": "Louder or quieter?"}
        with self._lock:
            cur = self._volume
        new = _gain.clamp_volume(round(cur + (VOLUME_STEP if d == "up" else -VOLUME_STEP), 4))
        self.set_volume(new)
        if abs(new - cur) < 1e-6:
            say = "That's as loud as I go." if d == "up" else "That's as quiet as it gets."
        else:
            say = "Louder." if d == "up" else "Quieter."
        return {"result": f"Volume {_gain.volume_to_percent(new)}%", "volume": new,
                "percent": _gain.volume_to_percent(new), "say": say}

    # ---------- favorites ----------
    def favorite(self, action: str, fav_id: str = "") -> dict[str, Any]:
        action = (action or "").strip().lower()
        if action == "save":
            with self._lock:
                item, track, artist = self._current_item, self._current_track, self._current_artist
            if not track:
                return {"error": "Nothing is playing", "say": "Nothing is playing to save."}
            vid = (item or {}).get("video_id", "")
            url = (item or {}).get("url", "") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
            added, entry = _fav.add(track, artist, vid, url)
            say = f"Saved {track} to your favorites." if added else f"{track} is already in your favorites."
            return {"result": "saved" if added else "already saved", "title": track, "say": say}
        favs = _fav.load()
        if action == "list":
            n = len(favs)
            if not n:
                return {"result": "none", "favorites": [], "say": "You don't have any favorites yet."}
            names = [f.get("title") or "?" for f in favs[-3:][::-1]]
            say = f"You have {n} favorites, including " + ", ".join(names) + "." if n > 3 else \
                  f"Your favorites: {', '.join(names)}."
            return {"result": f"{n} favorites", "favorites": favs, "say": say}
        if action == "play":
            if fav_id:
                favs = [f for f in favs if (f.get("id") or f.get("title")) == fav_id]
                if not favs:
                    return {"error": "No such favorite", "say": "I couldn't find that favorite."}
            if not favs:
                return {"error": "No favorites", "say": "You don't have any favorites yet."}
            random.shuffle(favs)
            n = self.enqueue_entries(favs)
            return {"result": f"Queued {n}", "say": f"Playing your {n} favorites." if n > 1 else f"Playing {favs[0].get('title') or 'your favorite'}."}
        if action == "remove":
            ok = _fav.remove(fav_id)
            return {"result": "removed"} if ok else {"error": "No such favorite", "say": "I couldn't find that favorite."}
        return {"error": f"Unknown action: {action}", "say": "I can't do that with favorites."}

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
            self._auto_active, self._auto_pool = False, []
            proc = self._detach_proc()
            self._state = "idle"
            self._clear_track()
            self._cond.notify_all()
        self._terminate_proc(proc)
        return {"result": "Music stopped", "say": "Okay, stopped."}

    def shutdown(self) -> None:
        """Stop playback so no ffplay is left behind (also registered with atexit)."""
        try:
            self.stop()
        except Exception:
            pass

    def _clear_track(self) -> None:
        """Called with the lock held."""
        self._current_track = ""
        self._current_item = None
        self._current_info = None
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
            item = None
            auto = False
            with self._cond:
                while True:
                    if self._state in ("idle", "loading") and self._proc is None:
                        if self._queue and not self._queue[0].get("pending"):
                            item = self._queue.popleft()
                            break
                        if not self._queue and self.autoplay and self._auto_active and self._auto_query:
                            auto = True
                            break
                    self._cond.wait(timeout=1.0)
                self._state = "loading"
                gen = self._gen

            if auto:
                item = self._auto_candidate(consume=True)
                with self._cond:
                    if item is None and gen == self._gen:
                        self._auto_active = False       # nothing new to play: stop continuing
                    if item is None or gen != self._gen or self._queue:
                        if gen == self._gen:
                            self._state = "idle"
                        continue

            label = item.get("title") or item.get("query") or "track"
            try:
                self._play_item(item, gen)
            except Exception as e:
                log.exception("Playback failed for: %s", label)
                with self._lock:
                    if gen == self._gen:
                        self._error = f"Couldn't play '{label}': {_clean_err(e)}"
                        self._state = "idle"
                        self._clear_track()
            with self._lock:
                if gen == self._gen and self._error:
                    self._fail_streak += 1
                    if self._fail_streak >= 3:          # don't spin forever on a broken autoplay run
                        self._auto_active = False

    # ---------- yt-dlp ----------
    @staticmethod
    def _extract(query: str) -> dict[str, Any]:
        """Resolve a query/URL to a stream URL with yt-dlp. Streaming only: nothing is downloaded."""
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
            "id": info.get("id") or "",
        }

    @staticmethod
    def _search(query: str, n: int) -> list[dict[str, Any]]:
        """One flat yt-dlp search (no stream resolution): [{id, title, artist, duration, url, live}]."""
        import yt_dlp

        opts = {
            "quiet": True, "no_warnings": True, "skip_download": True, "extract_flat": "in_playlist",
            "socket_timeout": 15, "extractor_retries": 1,
        }
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch{int(n)}:{query}", download=False)
        out = []
        for e in (info or {}).get("entries") or []:
            if not e:
                continue
            vid = e.get("id") or ""
            url = e.get("url") or e.get("webpage_url") or (f"https://www.youtube.com/watch?v={vid}" if vid else "")
            if not url:
                continue
            if not url.startswith("http") and vid:
                url = f"https://www.youtube.com/watch?v={vid}"
            try:
                dur = float(e.get("duration") or 0)
            except (TypeError, ValueError):
                dur = 0.0
            out.append({"id": vid, "title": e.get("title") or "", "artist": e.get("uploader") or e.get("channel") or "",
                        "duration": dur, "url": url,
                        "live": bool(e.get("is_live")) or e.get("live_status") == "is_live"})
        return out

    # ---------- prefetch / autoplay ----------
    def _kick_prefetch(self) -> None:
        """Resolve the next item's stream info in the background (yt-dlp takes seconds on the robot CPU)."""
        with self._lock:
            if self._prefetching:
                self._prefetch_again = True
                return
            self._prefetching = True
            self._prefetch_again = False
        threading.Thread(target=self._prefetch_loop, daemon=True, name="music-prefetch").start()

    def _prefetch_loop(self) -> None:
        while True:
            try:
                self._prefetch_once()
            except Exception as e:
                log.debug("prefetch failed: %s", e)
            with self._lock:
                if not self._prefetch_again:
                    self._prefetching = False
                    return
                self._prefetch_again = False

    def _prefetch_once(self) -> None:
        with self._lock:
            if self._state not in ("playing", "paused"):
                return
            head = self._queue[0] if self._queue else None
            if head is not None and head.get("pending"):
                return
            want_auto = head is None and self.autoplay and self._auto_active and bool(self._auto_query)
        if head is None:
            if not want_auto:
                return
            head = self._auto_candidate(consume=False)
        if head is None or self._fresh(head) or head.get("info_fail"):
            return
        try:
            info = self._extract(head.get("url") or head["query"])
        except Exception as e:
            head["info_fail"] = True
            log.debug("prefetch resolve failed for %s: %s", head.get("query"), e)
            return
        with self._lock:
            head["info"], head["info_ts"] = info, time.time()

    def _auto_candidate(self, consume: bool) -> dict[str, Any] | None:
        """Next autoplay track related to the last request; searches (outside the lock) when the pool is empty."""
        for _ in range(3):
            with self._lock:
                recent = set(self._recent)
                self._auto_pool = [i for i in self._auto_pool if i.get("video_id") not in recent]
                if self._auto_pool:
                    return self._auto_pool.pop(0) if consume else self._auto_pool[0]
                query, n = self._auto_query, self._auto_n
                if not query or self._auto_done:
                    return None
            try:
                entries = self._filter_entries(self._search(query, n), query)
            except Exception as e:
                log.warning("Autoplay search failed for %r: %s", query, e)
                return None
            with self._lock:
                if self._auto_query != query:
                    continue
                seen = set(self._recent) | {i.get("video_id") for i in self._queue}
                if self._current_item:
                    seen.add(self._current_item.get("video_id"))
                self._auto_pool = [self._new_item(query, "autoplay", e) for e in entries
                                   if not (e.get("id") and e["id"] in seen)]
                if n >= 40:
                    self._auto_done = True
                else:
                    self._auto_n = min(40, n * 2)
        return None

    # ---------- playback ----------
    def _play_query(self, query: str, gen: int) -> None:
        self._play_item(self._new_item(query), gen)

    def _play_item(self, item: dict[str, Any], gen: int) -> None:
        with self._lock:
            self._error = ""
        start, paused = 0.0, False
        while True:
            if self._fresh(item):
                info = item["info"]
            else:
                info = self._extract(item.get("url") or item["query"])
                item["info"], item["info_ts"] = info, time.time()
            item["title"] = item.get("title") or info["title"]
            item["artist"] = item.get("artist") or info["artist"]
            item["video_id"] = item.get("video_id") or info.get("id") or ""
            seek = self._run_pipeline(item, info, gen, start, paused)
            if seek is None:
                return
            start, paused = seek

    def _run_pipeline(self, item, info, gen, start, paused):
        """Play one track from `start` seconds. Returns (new_start, paused) if a seek restarted it, else None."""
        stream_url = info["url"]
        headers = info["headers"]
        title = info["title"]

        with self._lock:
            if gen != self._gen:
                return None  # skipped/stopped while the lookup was in flight
            self._current_track = title
            self._current_artist = info["artist"]
            self._current_item = item
            self._current_info = info
            self._duration = info["duration"]
            self._is_live = info["live"]
            self._start_time = time.time()
            self._elapsed_paused = float(start)
            self._seek_req = None
            self._state = "playing"
            self._fail_streak = 0
            if item.get("video_id") and start == 0:
                self._recent.append(item["video_id"])

            # Decode with ffmpeg to raw PCM, apply live gain in-process, play via
            # ffplay reading stdin. Volume/duck changes therefore act mid-track.
            # Streaming only: nothing is written to disk.
            decoder = None
            try:
                decoder = subprocess.Popen(
                    _gain.decoder_command(stream_url, headers, start),
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,  # loglevel warning: little output
                )
                _gain.shrink_pipe(decoder.stdout, 16384)
                proc = subprocess.Popen(
                    _gain.sink_command(), stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,  # binary stdin: raw PCM
                )
            except Exception as e:
                if decoder is not None:
                    try:
                        decoder.kill()
                    except Exception:
                        pass
                self._state = "idle"
                self._clear_track()
                self._error = f"Couldn't start ffmpeg/ffplay: {_clean_err(e)}"
                return None
            self._proc = proc
            _gain.shrink_pipe(proc.stdin, 16384)
            if paused:
                try:
                    proc.send_signal(signal.SIGSTOP)
                    self._state = "paused"
                except Exception:
                    pass
            # lock is held here: pass the initial gain explicitly (get_target re-locks)
            pump = _gain.GainPump(
                decoder.stdout, proc.stdin, self._target_gain,
                initial_gain=self._volume * self._duck_factor,
            )
            log.info("Starting playback for %s from %.0fs (gain=%.2fx)", title, start, pump.gain)
        pump_thread = threading.Thread(target=pump.run, daemon=True, name="music-gain")
        pump_thread.start()
        self._kick_prefetch()

        err = ""
        try:
            err = proc.stderr.read().decode("utf-8", "replace")
            proc.wait()
            if err and "error" in err.lower():
                log.warning("ffplay note: %s", err[:200])
        except Exception:
            pass
        decoder_rc = decoder.poll()          # None: still running (sink ended or was stopped first)
        try:
            decoder.kill()
            decoder.wait(timeout=1.0)
        except Exception:
            pass
        if decoder_rc not in (0, None):
            try:
                derr = decoder.stderr.read().decode("utf-8", "replace").strip()
            except Exception:
                derr = ""
            err = (err + "\n" + derr).strip() if derr else (err or f"ffmpeg exited with code {decoder_rc}")
            if proc.returncode == 0:
                proc.returncode = decoder_rc  # report it below as a failed track

        with self._lock:
            if self._proc is proc:
                self._proc = None
            if gen != self._gen:
                return None  # skip/stop already updated state
            if self._seek_req is not None:
                target, self._seek_req = self._seek_req, None
                return target, self._seek_paused
            if proc.returncode not in (0, None):
                self._error = f"Playback of '{title}' failed: {(err or 'ffplay error').strip()[-200:]}"
            self._state = "idle"
            self._clear_track()
        return None

    # ---------- seek ----------
    def seek(self, to: str) -> dict[str, Any]:
        """Jump inside the current song by restarting the same resolved stream with ffmpeg -ss."""
        with self._lock:
            state, info, proc = self._state, self._current_info, self._proc
            live, dur = self._is_live, self._duration
            if self._state == "playing":
                cur = self._elapsed_paused + (time.time() - self._start_time)
            else:
                cur = self._elapsed_paused
        if state not in ("playing", "paused") or info is None or proc is None:
            return {"error": "Nothing is playing", "say": "Nothing is playing."}
        if live:
            return {"error": "Live stream", "say": "I can't skip around in a live stream."}
        parsed = parse_seek(to)
        if parsed is None:
            return {"error": "Could not parse time", "say": "Skip to when? Say a time like one thirty."}
        mode, secs = parsed
        target = secs if mode == "abs" else cur + secs
        if dur and target >= dur - 1:
            res = self.skip()
            res["say"] = "That's past the end, skipping to the next song."
            return res
        target = max(0.0, target)
        if mode == "abs":
            say = "Starting the song over." if target == 0 else f"Skipping to {_fmt_time(target)}."
        elif target == 0 and secs < 0:
            say = "Starting the song over."
        else:
            say = f"{'Skipping ahead' if secs > 0 else 'Going back'} {_dur_words(abs(secs))}."
        with self._cond:
            if self._proc is not proc:       # track ended / changed while we were parsing
                return {"error": "Track changed", "say": "Sorry, that song just changed."}
            self._seek_req = target
            self._seek_paused = state == "paused"
            self._state = "loading"
            self._elapsed_paused = target
            self._proc = None
        self._terminate_proc(proc)
        return {"result": f"Seeked to {_fmt_time(target)}", "position": target, "say": say}


_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
              "ten": 10, "fifteen": 15, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
              "ninety": 90}
_UNIT_S = {"h": 3600, "hr": 3600, "hrs": 3600, "hour": 3600, "hours": 3600,
           "m": 60, "min": 60, "mins": 60, "minute": 60, "minutes": 60,
           "s": 1, "sec": 1, "secs": 1, "second": 1, "seconds": 1}


def _fmt_time(s: float) -> str:
    s = int(round(s))
    return f"{s // 60}:{s % 60:02d}"


def _dur_words(s: float) -> str:
    s = int(round(s))
    m, sec = divmod(s, 60)
    parts = []
    if m:
        parts.append(f"{m} minute{'s' if m != 1 else ''}")
    if sec or not m:
        parts.append(f"{sec} second{'s' if sec != 1 else ''}")
    return " ".join(parts)


def parse_seek(text: str) -> tuple[str, float] | None:
    """Parse free text into ('abs', seconds) or ('rel', signed seconds); None if no time is found."""
    t = re.sub(r"\s+", " ", str(text or "").strip().lower())
    if not t:
        return None
    if re.search(r"\b(start|beginning|begin|top|over|restart|zero)\b", t) and not re.search(r"\d", t):
        return ("abs", 0.0)
    t = re.sub(r"\b(" + "|".join(_NUM_WORDS) + r")\b", lambda m: str(_NUM_WORDS[m.group(1)]), t)
    t = t.replace("half a minute", "30 seconds").replace("half minute", "30 seconds")
    t = re.sub(r"\ban? (hour|minute|second)\b", r"1 \1", t)
    sign = 0
    if re.match(r"^\s*-", t) or re.search(r"\b(back|backward|backwards|rewind|earlier|behind|before)\b", t):
        sign = -1
    elif re.match(r"^\s*\+", t) or re.search(r"\b(forward|forwards|ahead|fast ?forward|later|further|more)\b", t):
        sign = 1
    secs = None
    m = re.search(r"(\d+):(\d{1,2})(?::(\d{1,2}))?", t)
    if m:
        a, b, c = int(m.group(1)), int(m.group(2)), m.group(3)
        secs = a * 3600 + b * 60 + int(c) if c is not None else a * 60 + b
    else:
        m = re.search(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m)\s*(?:and\s*)?(\d+)\b(?!\s*[a-z])", t)
        if m:
            secs = float(m.group(1)) * _UNIT_S[m.group(2)] + int(m.group(3))
        else:
            units = re.findall(r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|h|minutes?|mins?|m|seconds?|secs?|s)\b", t)
            if units:
                secs = sum(float(n) * _UNIT_S[u] for n, u in units)
            else:
                m = re.search(r"\d+(?:\.\d+)?", t)
                if m:
                    secs = float(m.group(0))
    if secs is None:
        return None
    if sign:
        return ("rel", sign * float(secs))
    return ("abs", float(secs))
