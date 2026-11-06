"""Offline tests for MusicPlayer playback/queue/process handling (yt_dlp and ffplay are mocked)."""
import subprocess
import sys
import threading
import time
import types

import pytest

from reachy_gemini_dj import music as music_mod
from reachy_gemini_dj.music import MusicPlayer


class _Sink:
    def write(self, b): return len(b)
    def flush(self): pass
    def close(self): pass
    def fileno(self): raise OSError("fake")


class FakeDecoder:
    def __init__(self, rc=0, err=b""):
        import io
        self.stdout = io.BytesIO(b"")
        self.stderr = io.BytesIO(err)
        self.rc = rc
        self.returncode = None
    def poll(self): return self.rc
    def kill(self): self.returncode = self.rc if self.rc is not None else -9
    def wait(self, timeout=None): return self.returncode


class FakeProc:
    def __init__(self, run_for=None, returncode=0, stderr=""):
        self._done = threading.Event()
        self._rc = returncode
        self.returncode = None
        self._stderr = stderr.encode() if isinstance(stderr, str) else stderr
        self.stderr = types.SimpleNamespace(read=self._read_err)
        self.stdin = _Sink()
        self.signals = []
        self.stopped = False
        if run_for is not None:
            threading.Timer(run_for, self._finish).start()

    def _finish(self, rc=None):
        if self.returncode is None:
            self.returncode = self._rc if rc is None else rc
        self._done.set()

    def poll(self):
        return self.returncode

    def send_signal(self, sig):
        import signal
        self.signals.append(sig)
        if sig == signal.SIGSTOP:
            self.stopped = True
        elif sig == signal.SIGCONT:
            self.stopped = False

    def terminate(self):
        self.signals.append("TERM")
        if not self.stopped:
            self._finish(-15)

    def kill(self):
        self.signals.append("KILL")
        self._finish(-9)

    def wait(self, timeout=None):
        if not self._done.wait(timeout):
            raise subprocess.TimeoutExpired("ffplay", timeout)
        return self.returncode

    def _read_err(self):
        self._done.wait()
        return self._stderr

    def communicate(self):
        self._done.wait()
        return "", self.stderr


def wait_for(pred, timeout=3.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def env(monkeypatch):
    procs = []
    ydl = types.SimpleNamespace(calls=[], info={}, delay=0.0, exc=None, gate=None)

    class FakeYDL:
        def __init__(self, opts):
            ydl.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, q, download=True):
            ydl.calls.append((q, download))
            if ydl.gate:
                ydl.gate.wait(5)
            if ydl.exc:
                raise ydl.exc
            info = ydl.info(q) if callable(ydl.info) else ydl.info
            return info

    monkeypatch.setitem(sys.modules, "yt_dlp", types.SimpleNamespace(YoutubeDL=FakeYDL))
    ydl.next_proc = lambda: FakeProc()

    ydl.decoders = []
    ydl.decoder_rc, ydl.decoder_err = 0, b""

    def popen(cmd, **kw):
        if cmd[0] == "ffmpeg":
            d = FakeDecoder(ydl.decoder_rc, ydl.decoder_err)
            d.cmd = cmd
            ydl.decoders.append(d)
            return d
        p = ydl.next_proc()
        p.cmd = cmd
        procs.append(p)
        return p

    monkeypatch.setattr(music_mod.subprocess, "Popen", popen)
    monkeypatch.setattr(music_mod._gain, "_ffplay_major", lambda: 8)
    ydl.procs = procs
    ydl.info = lambda q: {"url": "http://s/" + q, "title": q, "uploader": "u", "duration": 100}
    players = []

    def make():
        p = MusicPlayer(0.5)
        players.append(p)
        return p

    ydl.make = make
    yield ydl
    for p in players:
        p.stop()


def test_play_empty_query(env):
    p = env.make()
    assert "error" in p.play("   ")
    assert p.get_status()["state"] == "idle"


def test_streams_not_downloads(env):
    p = env.make()
    p.play("song")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    assert env.calls == [("song", False)]
    assert "http://s/song" in env.decoders[0].cmd          # ffmpeg reads the stream URL directly
    assert env.procs[0].cmd[-1] == "pipe:0"                 # ffplay only plays piped PCM


def test_natural_end_advances_queue(env):
    env.next_proc = lambda: FakeProc(run_for=0.1)
    p = env.make()
    p.play("a")
    p.play("b")
    assert wait_for(lambda: [c[0] for c in env.calls] == ["a", "b"])
    assert wait_for(lambda: p.get_status()["state"] == "idle" and p.get_status()["track"] == "")


def test_livestream_none_duration(env):
    env.info = lambda q: {"url": "http://live", "title": "L", "duration": None, "is_live": True}
    p = env.make()
    p.play("live")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    st = p.get_status()
    assert st["duration"] == 0.0 and st["is_live"] is True and st["error"] == ""


def test_playlist_entries_first_nonempty(env):
    env.info = lambda q: {"entries": [None, {"url": "http://x", "title": "T"}]}
    p = env.make()
    p.play("q")
    assert wait_for(lambda: p.get_status()["track"] == "T")


def test_empty_results_clean_error(env):
    env.info = lambda q: {"entries": []}
    p = env.make()
    p.play("nothing")
    assert wait_for(lambda: p.get_status()["error"])
    st = p.get_status()
    assert st["state"] == "idle" and "no results" in st["error"]


def test_ytdlp_exception_does_not_kill_worker(env):
    env.exc = RuntimeError("\x1b[0;31mERROR:\x1b[0m boom")
    p = env.make()
    p.play("bad")
    assert wait_for(lambda: "boom" in p.get_status()["error"])
    assert "\x1b" not in p.get_status()["error"]
    env.exc = None
    p.play("good")
    assert wait_for(lambda: p.get_status()["track"] == "good")
    assert p.get_status()["error"] == ""


def test_format_fallback_url(env):
    env.info = lambda q: {"title": "F", "formats": [{"url": "http://v", "acodec": "none"}, {"url": "http://a", "acodec": "opus"}]}
    p = env.make()
    p.play("q")
    assert wait_for(lambda: env.procs)
    assert "http://a" in env.decoders[0].cmd


def test_pause_resume_signals_and_elapsed(env):
    import signal
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    assert p.pause() == {"state": "paused"}
    proc = env.procs[0]
    assert proc.signals == [signal.SIGSTOP]
    e1 = p.get_status()["elapsed"]
    time.sleep(0.1)
    assert p.get_status()["elapsed"] == e1
    assert p.resume() == {"state": "playing"}
    assert proc.signals[-1] == signal.SIGCONT


def test_pause_resume_when_idle_return_error(env):
    p = env.make()
    assert "error" in p.pause()
    assert "error" in p.resume()
    assert "error" in p.skip()


def test_skip_paused_process_is_continued_then_terminated(env):
    import signal
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    p.pause()
    p.skip()
    proc = env.procs[0]
    assert signal.SIGCONT in proc.signals and "TERM" in proc.signals
    assert proc.returncode is not None
    assert p.get_status()["state"] == "idle"


def test_skip_plays_next_in_queue(env):
    p = env.make()
    p.play("a")
    p.play("b")
    assert wait_for(lambda: p.get_status()["track"] == "a")
    p.skip()
    assert wait_for(lambda: p.get_status()["track"] == "b")
    assert env.procs[0].returncode is not None


def test_skip_during_lookup_discards_track(env):
    env.gate = threading.Event()
    p = env.make()
    p.play("a")
    assert wait_for(lambda: env.calls)
    assert p.get_status()["state"] == "loading"
    p.skip()
    env.gate.set()
    time.sleep(0.3)
    assert env.procs == []
    assert p.get_status()["state"] == "idle"


def test_stop_during_lookup_discards_track_and_queue(env):
    env.gate = threading.Event()
    p = env.make()
    p.play("a")
    p.play("b")
    assert wait_for(lambda: env.calls)
    p.stop()
    env.gate.set()
    time.sleep(0.3)
    assert env.procs == [] and [c[0] for c in env.calls] == ["a"]
    st = p.get_status()
    assert st["state"] == "idle" and st["queued"] == 0


def test_stop_clears_everything_and_status(env):
    p = env.make()
    p.play("a")
    p.play("b")
    assert wait_for(lambda: p.get_status()["track"] == "a")
    p.stop()
    st = p.get_status()
    assert st["state"] == "idle" and st["track"] == "" and st["queued"] == 0
    time.sleep(0.2)
    assert len(env.procs) == 1


def test_ffplay_missing_clean_error(env, monkeypatch):
    def boom(cmd, **kw):
        raise FileNotFoundError("ffplay")
    monkeypatch.setattr(music_mod.subprocess, "Popen", boom)
    p = env.make()
    p.play("a")
    assert wait_for(lambda: "ffplay" in p.get_status()["error"])
    assert p.get_status()["state"] == "idle"


def test_ffplay_nonzero_exit_reports_error(env):
    env.next_proc = lambda: FakeProc(run_for=0.05, returncode=1, stderr="Server returned 403 Forbidden")
    p = env.make()
    p.play("a")
    assert wait_for(lambda: "403" in p.get_status()["error"])
    assert p.get_status()["state"] == "idle"


def test_elapsed_clamped_to_duration(env):
    env.info = lambda q: {"url": "u", "title": "t", "duration": 0.05}
    p = env.make()
    p.play("a")
    assert wait_for(lambda: p.get_status()["state"] == "playing")
    time.sleep(0.2)
    assert p.get_status()["elapsed"] <= 0.05


def test_decoder_failure_reports_error(env):
    env.decoder_rc, env.decoder_err = 8, b"Server returned 403 Forbidden"
    env.next_proc = lambda: FakeProc(run_for=0.05, returncode=0)
    p = env.make()
    p.play("a")
    assert wait_for(lambda: "403" in p.get_status()["error"])
    assert p.get_status()["state"] == "idle"
