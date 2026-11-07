"""Gamepad drive mode (motion.py): clamping, speed-capped stepping, pose math and mode interplay, with a fake robot."""
import math
import random
import threading
import time

import numpy as np
import pytest

from reachy_gemini_dj import motion as mo
from reachy_gemini_dj import poses as poses_mod
from reachy_gemini_dj.poses import PoseStore


class FakeMini:
    def __init__(self):
        self.calls = []
        self.targets = []
        self.head = mo.head_pose(0, 0, 5, 0, 10, 20)
    def enable_motors(self, ids=None): self.calls.append("motors_on")
    def disable_motors(self, ids=None): self.calls.append("motors_off")
    def get_current_head_pose(self): return self.head.copy()
    def get_current_joint_positions(self): return ([math.radians(30), 0, 0, 0, 0, 0, 0], [math.radians(10), math.radians(-20)])
    def set_target(self, head=None, antennas=None, body_yaw=None):
        self.targets.append((head, antennas, body_yaw))
    def goto_target(self, head=None, antennas=None, duration=0.5, body_yaw=0.0, **k):
        self.calls.append("goto")
    def play_move(self, *a, **k): self.calls.append("play")


@pytest.fixture
def motion(tmp_path, monkeypatch):
    monkeypatch.setattr(poses_mod, "POSES_FILE", tmp_path / "poses.json")
    stop = threading.Event()
    m = mo.Motion(FakeMini(), stop)
    m.poses = PoseStore(tmp_path / "poses.json")
    yield m
    stop.set()


def test_rpy_round_trip():
    for _ in range(300):
        r, p, y = random.uniform(-25, 25), random.uniform(-30, 30), random.uniform(-60, 60)
        back = mo.matrix_to_rpy(mo.head_pose(1, 2, 3, r, p, y))
        assert back == pytest.approx((r, p, y), abs=1e-9)


def test_head_pose_translation_is_mm():
    m = mo.head_pose(10, -5, 20, 0, 0, 0)
    assert m[:3, 3] == pytest.approx([0.010, -0.005, 0.020])
    assert np.allclose(m[:3, :3], np.eye(3))


def test_clamp_drops_junk_and_limits():
    out = mo.drive_clamp({"yaw": 500, "pitch": "-99", "roll": float("nan"), "z": None, "bogus": 3, "ant_l": "x"})
    assert out == {"yaw": 60.0, "pitch": -30.0}


def test_step_is_speed_capped():
    cur = {k: 0.0 for k in mo.DRIVE_KEYS}
    nxt = mo.drive_step(cur, {"yaw": 60.0, "x": -15.0}, 0.02)
    assert nxt["yaw"] == pytest.approx(mo.DRIVE_SPEED["yaw"] * 0.02)
    assert nxt["x"] == pytest.approx(-mo.DRIVE_SPEED["x"] * 0.02)
    assert nxt["pitch"] == 0.0
    near = mo.drive_step({"yaw": 59.9}, {"yaw": 60.0}, 0.02)
    assert near["yaw"] == pytest.approx(60.0)        # never overshoots


def test_start_reads_present_pose(motion):
    st = motion.drive_start()
    assert st["manual"] is True and motion.manual
    t = st["target"]
    assert t["pitch"] == pytest.approx(10) and t["yaw"] == pytest.approx(20) and t["z"] == pytest.approx(5)
    assert t["body"] == pytest.approx(30)
    assert t["ant_r"] == pytest.approx(10) and t["ant_l"] == pytest.approx(-20)
    assert "motors_on" in motion.mini.calls
    assert motion.moving()                            # the IMU watcher ignores the sensor meanwhile
    motion.drive_stop()


def test_antenna_limits_match_pollens_range():
    # Pollen's recorded moves keep the left antenna >= -60 deg and the right one <= +60 deg
    assert mo.DRIVE_LIMITS["ant_l"][0] >= -60 and mo.DRIVE_LIMITS["ant_r"][1] <= 60


def test_start_outside_range_walks_back_without_a_jump(motion):
    motion.mini.get_current_joint_positions = lambda: ([0.0] * 7, [0.0, math.radians(-100)])   # left antenna past its stop
    st = motion.drive_start()
    assert st["target"]["ant_l"] == mo.DRIVE_LIMITS["ant_l"][0]
    time.sleep(0.1)
    motion.drive_stop()
    first = motion.mini.targets[0][1]                 # [right, left] in radians
    assert math.degrees(first[1]) < mo.DRIVE_LIMITS["ant_l"][0] - 30   # first command is near -100, not the limit


def test_set_requires_mode(motion):
    with pytest.raises(RuntimeError):
        motion.drive_set({"yaw": 10})


def test_loop_streams_towards_target(motion):
    motion.drive_start()
    motion.drive_set({"yaw": -60, "pitch": 0})
    time.sleep(0.4)
    motion.drive_stop()
    assert motion.mini.targets, "the drive loop sent nothing"
    head, ants, body = motion.mini.targets[-1]
    assert head.shape == (4, 4) and len(ants) == 2 and isinstance(body, float)
    _, _, yaw = mo.matrix_to_rpy(head)
    assert -60 <= yaw < 20                            # moving down from 20 at <= 120 deg/s, not jumped to -60
    assert yaw > 20 - mo.DRIVE_SPEED["yaw"] * 0.6


def test_other_moves_pause_while_driving(motion):
    motion.drive_start()
    motion.move_head("left")
    motion.thinking(True)
    time.sleep(0.2)
    assert "goto" not in motion.mini.calls
    motion.drive_stop()
    assert not motion.manual


def test_learn_and_pose_play_leave_drive_mode(motion):
    motion.drive_start()
    motion.learn_start()
    assert not motion.manual and motion.learning
    motion.learn_stop()
    motion.drive_start()
    motion.poses.save("wave", {"head": np.eye(4).tolist(), "antennas": [0, 0], "head_joints": [0] * 7})
    motion.goto_pose("wave")
    assert not motion.manual


def test_idle_timeout_leaves_mode(motion, monkeypatch):
    monkeypatch.setattr(mo, "DRIVE_IDLE_S", 0.1)
    motion.drive_start()
    time.sleep(0.4)
    assert not motion.manual
