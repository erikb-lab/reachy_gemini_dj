"""Push-to-talk gating, the audio pipeline's talk API, and the thinking head tilt (fakes only, no hardware)."""
import sys
import threading
import time
import types
from types import SimpleNamespace

import numpy as np
import pytest

from reachy_gemini_dj.audio import AudioPipeline, PttGate

FR = 320                                  # 20 ms of 16 kHz audio, in samples
LOUD = (np.ones(FR) * 3000).astype("<i2")


def frame(tag: int = 1) -> bytes:
    return (np.ones(FR, dtype="<i2") * tag).tobytes()


class Clock:
    def __init__(self):
        self.t = 100.0

    def __call__(self):
        return self.t


# ------------------------------------------------------------------ PttGate ---
def test_frames_dropped_unless_held():
    g = PttGate()
    assert all(g.process(frame(i)) == [] for i in range(1, 6))
    g.press()
    out = g.process(frame(9))
    assert out[-1] == frame(9)
    g.release()
    assert g.process(frame(7)) == [bytes(FR * 2)]       # silence tail, not the mic


def test_preroll_delivered_on_press_and_capped():
    g = PttGate(preroll_s=0.3)
    for i in range(1, 41):                              # 0.8 s of idle audio, tagged 1..40
        g.process(frame(i))
    assert g.press() is True
    out = g.process(frame(99))
    assert out[-1] == frame(99)
    pre = out[:-1]
    assert 0.3 <= len(pre) * 0.02 <= 0.32 + 1e-9        # ~300 ms, not the whole 0.8 s
    assert pre[-1] == frame(40)                         # most recent audio, in order
    assert [np.frombuffer(p, "<i2")[0] for p in pre] == sorted(np.frombuffer(p, "<i2")[0] for p in pre)
    assert g.process(frame(100)) == [frame(100)]        # pre-roll is delivered only once


def test_silence_tail_length_then_idle():
    g = PttGate(tail_s=0.9)
    g.press(); g.process(frame())
    g.release()
    sent = 0
    for _ in range(100):
        out = g.process(frame(5))
        if not out:
            break
        assert out == [bytes(FR * 2)]
        sent += len(out[0])
    assert abs(sent / 2 / 16000 - 0.9) < 0.021          # ~0.9 s of silence, paced one frame per frame
    assert g.process(frame(5)) == []                    # back to dropping


def test_release_clears_stale_preroll():
    g = PttGate()
    g.process(frame(1))
    g.press(); g.process(frame(2)); g.release()
    for _ in range(60):                                 # run out the tail, refill pre-roll with new audio
        g.process(frame(3))
    g.press()
    assert set(np.frombuffer(b"".join(g.process(frame(4))), "<i2")) <= {3, 4}   # nothing from before the last press


def test_auto_release_after_30s():
    c = Clock()
    g = PttGate(clock=c)
    g.press()
    assert g.active
    c.t += 29
    assert g.active and g.process(frame(1)) == [frame(1)]
    c.t += 2
    assert not g.active
    assert g.process(frame(1)) == [bytes(FR * 2)]       # tail follows the safety release


def test_toggle_and_double_press():
    g = PttGate()
    assert g.toggle() is True and g.active
    assert g.press() is False                           # already held: not a new press
    assert g.toggle() is False and not g.active
    assert g.release() is False


# ------------------------------------------------------- AudioPipeline wiring ---
class FakeMusic:
    is_playing = False
    _volume = 1.0

    def __init__(self):
        self.ducks = []

    def set_duck(self, v): self.ducks.append(("duck", v))
    def reset_duck(self): self.ducks.append(("reset",))


class FakeMedia:
    """Feeds scripted mic frames, then blocks until stopped."""
    def __init__(self, frames):
        self.frames = list(frames)

    def get_audio_sample(self):
        return self.frames.pop(0) if self.frames else None


def make_pipe(mode, frames=(), on_pressed=None):
    cfg = SimpleNamespace(listen_mode=mode, duck_level=0.2, echo_guard=1.5, voice_gain=1.0, hw_speaker_percent=0)
    got = []
    media = FakeMedia(frames)
    stop = threading.Event()
    p = AudioPipeline(SimpleNamespace(media=media), stop, cfg, FakeMusic(), on_pcm_chunk=got.append,
                      on_talk_pressed=on_pressed)
    p.in_sr = 16000
    return p, got, stop


def run_loop(p, stop, n_frames_wait=0.3):
    t = threading.Thread(target=p._capture_loop, daemon=True)
    t.start()
    time.sleep(n_frames_wait)
    stop.set()
    t.join(2)


def test_always_mode_unchanged_sends_every_frame():
    frames = [LOUD.copy() for _ in range(5)]
    p, got, stop = make_pipe("always", frames)
    run_loop(p, stop)
    assert len(got) == 5 and not p.ptt_active
    p.ptt_set(True)
    assert not p.ptt_active                             # no gating state in always mode


def test_ptt_mode_gates_mic_end_to_end():
    quiet = [(np.ones(FR) * 50).astype("<i2") for _ in range(10)]
    p, got, stop = make_pipe("push_to_talk", quiet)
    run_loop(p, stop)
    assert got == []                                    # nothing leaves the robot until talk is held

    p2, got2, stop2 = make_pipe("push_to_talk", [])
    p2.ptt_set(True)
    assert p2.ptt_active
    p2.media.frames.extend(LOUD.copy() for _ in range(5))
    run_loop(p2, stop2)
    assert len(got2) == 5 and all(abs(int(np.frombuffer(c, "<i2")[0]) - 3000) <= 1 for c in got2)   # unattenuated (x0.05 would be ~150)
    assert p2.ptt_set(False) is False and p2._ptt._tail_left > 0


def test_press_barges_in_and_calls_hook():
    calls = []
    p, got, stop = make_pipe("push_to_talk", on_pressed=lambda: calls.append(1))
    p.play_voice(np.zeros(2400, dtype="<i2").tobytes())
    assert p._is_robot_speaking
    p.ptt_set(True)
    assert not p._is_robot_speaking and calls == [1]    # robot's speech cut, hook fired
    p.ptt_set(True)
    assert calls == [1]                                 # still held: not a new press
    p.ptt_set(False); p.ptt_toggle()
    assert calls == [1, 1] and p.ptt_active


def test_hook_and_barge_in_also_in_always_mode():
    calls = []
    p, _, _ = make_pipe("always", on_pressed=lambda: calls.append(1))
    p.play_voice(np.zeros(2400, dtype="<i2").tobytes())
    p.ptt_set(True)
    assert calls == [1] and not p._is_robot_speaking


def test_bad_mode_falls_back_to_always():
    p, _, _ = make_pipe("nonsense")
    assert p.listen_mode == "always"


# ------------------------------------------------------------- thinking tilt ---
@pytest.fixture
def fake_sdk(monkeypatch):
    mod = types.ModuleType("reachy_mini")
    utils = types.ModuleType("reachy_mini.utils")
    utils.create_head_pose = lambda **k: ("pose", tuple(sorted(k.items())))
    mod.utils = utils
    monkeypatch.setitem(sys.modules, "reachy_mini", mod)
    monkeypatch.setitem(sys.modules, "reachy_mini.utils", utils)


class FakeMini:
    def __init__(self):
        self.calls = []

    def enable_motors(self, ids=None): pass
    def disable_motors(self, ids=None): pass
    def goto_target(self, head=None, **k): self.calls.append((head, k.get("duration")))


@pytest.fixture
def motion(fake_sdk, tmp_path, monkeypatch):
    from reachy_gemini_dj import poses as poses_mod
    monkeypatch.setattr(poses_mod, "POSES_FILE", tmp_path / "poses.json")
    from reachy_gemini_dj.motion import Motion
    stop = threading.Event()
    en = {"on": True}
    m = Motion(FakeMini(), stop, enabled=lambda: en["on"])
    m.en = en
    yield m
    stop.set()


def settle(m, n):
    for _ in range(100):
        if len(m.mini.calls) >= n and not m.busy and m._q.empty():
            return
        time.sleep(0.02)


def test_thinking_tilts_then_levels(motion):
    motion.thinking(True)
    settle(motion, 1)
    assert len(motion.mini.calls) == 1 and motion._tilted
    roll = dict(motion.mini.calls[0][0][1])["roll"]
    assert 0 < roll <= 12 and motion.mini.calls[0][1] >= 1.0       # small and slow
    motion.thinking(True)                                           # repeated state: no second move
    motion.thinking(False)
    settle(motion, 2)
    assert len(motion.mini.calls) == 2 and not motion._tilted
    assert dict(motion.mini.calls[1][0][1]) == {"degrees": True} or dict(motion.mini.calls[1][0][1]) == {}
    motion.thinking(False)                                          # nothing to undo: no move
    time.sleep(0.1)
    assert len(motion.mini.calls) == 2


def test_thinking_skipped_when_disabled_learning_or_busy(motion):
    motion.en["on"] = False
    motion.thinking(True)
    assert not motion._thinking and motion.mini.calls == []
    motion.en["on"] = True
    motion.learning = True
    motion.thinking(True)
    assert not motion._thinking
    motion.learning = False
    motion.busy = True                                              # a dance / emotion is running
    motion.thinking(True)
    time.sleep(0.1)
    assert not motion._thinking and motion.mini.calls == []


def test_thinking_ends_before_tilt_runs_does_not_leave_head_tilted(motion):
    gate = threading.Event()
    motion._submit("blocker", lambda: gate.wait(1))                 # keeps the worker occupied
    time.sleep(0.1)
    motion.busy = False                                             # (thinking() only defers to a *queued* move)
    motion.thinking(True)
    motion.thinking(False)
    gate.set()
    time.sleep(0.4)
    assert not motion._tilted and motion.mini.calls == []           # queued tilt noticed the turn was over


class FakeMotion:
    def __init__(self):
        self.calls = []

    def thinking(self, on): self.calls.append(on)


def test_state_callback_drives_thinking():
    """RemoteSession reports brain states; the app maps them to motion.thinking (same mapping as main.py)."""
    from reachy_gemini_dj.remote_session import RemoteSession
    fm = FakeMotion()
    seen = []

    def on_state(state, detail=""):
        seen.append((state, detail))
        fm.thinking(state == "thinking")

    rs = RemoteSession(SimpleNamespace(), lambda n, a: {}, lambda k, t: None, on_state=on_state)
    for s in ("thinking", "speaking", "thinking", "listening"):
        rs._emit_state(s, "d")
    assert seen[0] == ("thinking", "d") and fm.calls == [True, False, True, False]

    def boom(s, d=""):
        raise RuntimeError("x")
    RemoteSession(SimpleNamespace(), lambda n, a: {}, lambda k, t: None, on_state=boom)._emit_state("thinking")  # swallowed
    RemoteSession(SimpleNamespace(), lambda n, a: {}, lambda k, t: None)._emit_state("thinking")                  # default no-op
