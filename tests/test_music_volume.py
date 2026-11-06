"""Offline tests for music volume / ducking (no audio, no network, no real subprocesses)."""
import io
import math
import sys
import time
import types

import numpy as np
import pytest

from reachy_gemini_dj import gain
from reachy_gemini_dj.gain import (
    MAX_VOLUME, GainPump, apply_gain, clamp_duck, clamp_volume,
    decoder_command, percent_to_volume, sink_command, slew, volume_to_percent,
)

BLOCK_SAMPLES = gain.SAMPLE_RATE * gain.BLOCK_MS // 1000 * gain.CHANNELS


def pcm(vals):
    return np.array(vals, dtype="<i2").tobytes()


def samples(b):
    return np.frombuffer(b, dtype="<i2")


def const_block(level=10000):
    return np.full(BLOCK_SAMPLES, level, dtype="<i2").tobytes()


class _Sink(io.BytesIO):
    def close(self):  # keep the buffer readable after the pump closes it
        self.closed_flag = True


# ---- clamping / scale ----
def test_clamp_volume():
    assert clamp_volume(-1) == 0.0
    assert clamp_volume(1.0) == 1.0
    assert clamp_volume(2.0) == 2.0
    assert clamp_volume(99) == MAX_VOLUME == 2.5
    assert clamp_volume(float("nan")) == gain.DEFAULT_VOLUME
    assert clamp_volume("abc") == gain.DEFAULT_VOLUME


def test_clamp_duck():
    assert clamp_duck(-0.5) == 0.0
    assert clamp_duck(5) == 1.0
    assert clamp_duck(0.2) == 0.2
    assert clamp_duck(float("nan")) == 1.0


def test_percent_mapping():
    assert percent_to_volume(100) == 1.0
    assert percent_to_volume(250) == 2.5
    assert percent_to_volume(1000) == 2.5
    assert percent_to_volume(-5) == 0.0
    assert volume_to_percent(2.0) == 200
    assert volume_to_percent(9) == 250


# ---- gain math ----
def test_unity_passthrough_and_silence():
    d = pcm([100, -100, 200, -200])
    assert apply_gain(d, 1.0, 1.0) == d
    assert samples(apply_gain(d, 0.0, 0.0)).tolist() == [0, 0, 0, 0]


def test_constant_gain_and_clipping():
    assert samples(apply_gain(pcm([1000, -1000, 20000, -20000]), 0.5, 0.5)).tolist() == [500, -500, 10000, -10000]
    assert samples(apply_gain(pcm([20000, -20000]), 2.5, 2.5)).tolist() == [32767, -32768]


def test_ramp_is_monotonic_and_ends_at_target():
    out = samples(apply_gain(np.full(200, 10000, dtype="<i2").tobytes(), 1.0, 0.2))
    left = out[0::2]
    assert (np.diff(left) <= 0).all()
    assert abs(int(left[-1]) - 2000) <= 1
    assert 9500 < left[0] < 10000  # no step at block start


def test_partial_frame_dropped():
    assert len(apply_gain(b"\x01\x00\x02", 0.5, 0.5)) == 0
    assert len(apply_gain(b"\x01\x00\x02\x00\x03\x00\x04\x00\x05", 0.5, 0.5)) == 8


def test_slew_attack_release():
    down = 1.0 - slew(1.0, 0.0, 0.02)
    up = slew(0.0, 1.0, 0.02)
    assert down > up  # attack faster than release
    assert math.isclose(down, 0.02 / gain.ATTACK_S)
    assert slew(0.5, 0.5, 0.02) == 0.5
    assert slew(0.21, 0.2, 0.02) == 0.2  # no overshoot
    assert slew(0.19, 0.2, 0.02) == 0.2


# ---- duck state machine through the pump ----
def test_pump_duck_attack_then_release():
    state = {"t": 1.0}
    pump = GainPump(io.BytesIO(), _Sink(), lambda: state["t"], initial_gain=1.0)
    block = const_block()
    state["t"] = 0.2
    downs = []
    for _ in range(10):  # 200 ms
        pump.process_block(block)
        downs.append(pump.gain)
    assert 0.2 < downs[0] < 1.0  # smooth, not a click
    assert downs == sorted(downs, reverse=True)
    assert downs[-1] == 0.2
    state["t"] = 1.0
    ups = []
    for _ in range(40):  # 800 ms
        pump.process_block(block)
        ups.append(pump.gain)
    assert ups[0] < 0.4  # slow release
    assert ups == sorted(ups)
    assert ups[-1] == 1.0
    assert downs.index(0.2) < ups.index(1.0)  # attack settles sooner than release


def test_pump_live_volume_change_applies_mid_stream():
    state = {"t": 1.0}
    pump = GainPump(io.BytesIO(), _Sink(), lambda: state["t"], initial_gain=1.0)
    out = []
    for i in range(40):
        if i == 20:
            state["t"] = 0.5
        out.append(int(samples(pump.process_block(const_block()))[-1]))
    assert out[19] == 10000
    assert out[20] < 10000  # change takes effect on the very next block
    assert out[-1] == 5000


def test_pump_run_copies_everything_and_closes():
    dst = _Sink()
    pump = GainPump(io.BytesIO(const_block() * 5), dst, lambda: 1.0)
    pump.run()
    assert len(dst.getvalue()) == 5 * pump.block_bytes
    assert dst.closed_flag


def test_pump_survives_broken_pipe():
    class Broken(io.BytesIO):
        def write(self, b):
            raise BrokenPipeError

    GainPump(io.BytesIO(const_block()), Broken(), lambda: 1.0).run()  # must not raise


# ---- command construction ----
def test_commands_stream_only():
    d = decoder_command("http://x/a", {"UA": "z"})
    assert d[0] == "ffmpeg" and "pipe:1" in d and "s16le" in d and "-headers" in d
    assert "UA: z\r\n" in d
    s = sink_command()
    assert s[0] == "ffplay" and "pipe:0" in s and "-volume" not in s


# ---- MusicPlayer ----
@pytest.fixture
def player(monkeypatch):
    from reachy_gemini_dj import music
    monkeypatch.setattr(music.MusicPlayer, "_queue_worker", lambda self: None)
    return music.MusicPlayer


def test_player_volume_and_duck_state(player):
    p = player(volume=2.0)
    assert p._target_gain() == 2.0  # config default no longer truncated to 1.0
    p.set_volume(5.0)
    assert p._volume == 2.5
    p.set_volume(-1)
    assert p._volume == 0.0
    p.set_volume(2.0)
    p.set_duck(0.2)
    assert math.isclose(p._target_gain(), 0.4)
    p.set_duck(7)
    assert p._duck_factor == 1.0
    p.set_duck(0.2)
    p.reset_duck()
    assert p._target_gain() == 2.0
    assert player(volume=9)._volume == 2.5


def test_play_query_wires_pipeline_without_real_processes(player, monkeypatch):
    from reachy_gemini_dj import music

    fake_ydl = types.ModuleType("yt_dlp")

    class YDL:
        def __init__(self, *a, **k): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def extract_info(self, q, download=False):
            return {"url": "http://stream", "title": "T", "uploader": "U", "duration": 1, "http_headers": {}}

    fake_ydl.YoutubeDL = YDL
    monkeypatch.setitem(sys.modules, "yt_dlp", fake_ydl)

    class Proc:
        def __init__(self, cmd, **kw):
            self.cmd = cmd
            self.stdout = io.BytesIO(const_block() * 4)
            self.stdin = _Sink()
            self.stderr = io.BytesIO(b"")

        returncode = 0
        def wait(self, timeout=None): return 0
        def poll(self): return 0
        def kill(self): pass
        def terminate(self): pass

    procs = []
    monkeypatch.setattr(music.subprocess, "Popen", lambda cmd, **kw: procs.append(Proc(cmd, **kw)) or procs[-1])
    monkeypatch.setattr(music._gain, "shrink_pipe", lambda *a, **k: None)
    monkeypatch.setattr(music._gain, "_ffplay_major", lambda: 8)

    p = player(volume=0.5)
    p._play_query("x", p._gen)
    assert procs[0].cmd[0] == "ffmpeg" and procs[1].cmd[0] == "ffplay"
    time.sleep(0.3)  # pump thread
    data = samples(procs[1].stdin.getvalue())
    assert len(data) == 4 * BLOCK_SAMPLES and (data == 5000).all()
    assert p.get_status()["state"] == "idle"
