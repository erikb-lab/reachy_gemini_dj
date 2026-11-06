"""Learn-pose storage and Motion learn mode with a fake robot."""
import json
import threading
import time

import numpy as np
import pytest

from reachy_gemini_dj import poses as poses_mod
from reachy_gemini_dj.poses import PoseStore


class FakeMini:
    def __init__(self):
        self.calls = []
        self.head = np.eye(4)
        self.head[0, 3] = 0.01
    def enable_gravity_compensation(self): self.calls.append("gc_on")
    def disable_gravity_compensation(self): self.calls.append("gc_off")
    def enable_motors(self, ids=None): self.calls.append("motors_on")
    def disable_motors(self, ids=None): self.calls.append("motors_off")
    def get_current_head_pose(self): return self.head.copy()
    def get_present_antenna_joint_positions(self): return [0.1, -0.2]
    def get_current_joint_positions(self): return ([0.3, 0, 0, 0, 0, 0, 0], [0.1, -0.2])
    def set_target(self, head=None, antennas=None, body_yaw=None): self.calls.append("hold")
    def goto_target(self, head=None, antennas=None, duration=0.5, body_yaw=0.0, **k):
        self.calls.append(("goto", float(head[0, 3]), list(antennas), body_yaw))


@pytest.fixture
def motion(tmp_path, monkeypatch):
    monkeypatch.setattr(poses_mod, "POSES_FILE", tmp_path / "poses.json")
    from reachy_gemini_dj.motion import Motion
    stop = threading.Event()
    m = Motion(FakeMini(), stop)
    m.poses = PoseStore(tmp_path / "poses.json")
    yield m
    stop.set()


def test_store_saves_to_disk_immediately_and_reloads(tmp_path):
    f = tmp_path / "p.json"
    s = PoseStore(f)
    assert s.save("  Big   Wave ", {"head": [[1]]}) == "Big Wave"
    assert json.loads(f.read_text())["Big Wave"]["head"] == [[1]]
    assert PoseStore(f).names() == ["Big Wave"]
    assert s.find("wave")[0] == "Big Wave"             # forgiving voice match
    s.save("big wave", {"head": [[2]]})                 # same name, any case: overwrite
    assert PoseStore(f).names() == ["big wave"]
    assert s.delete("BIG WAVE") and PoseStore(f).names() == []
    with pytest.raises(ValueError):
        s.save("   ", {"head": []})


def test_corrupt_file_is_tolerated(tmp_path):
    f = tmp_path / "p.json"
    f.write_text("{not json")
    assert PoseStore(f).names() == []


def test_learn_save_play_cycle(motion):
    assert "motors off" in motion.learn_start()
    assert motion.learning and motion.mini.calls[-1] == "motors_off"
    motion.move_head("left")                            # ignored while the motors are off
    name = motion.save_pose("wave")                     # locks first, then saves
    assert name == "wave" and motion.poses.names() == ["wave"]
    assert not motion.learning and motion.mini.calls[-3:] == ["hold", "motors_on", "hold"]
    motion.mini.head = np.eye(4)
    assert motion.goto_pose("WAVE") == "wave"
    for _ in range(50):
        if any(isinstance(c, tuple) for c in motion.mini.calls):
            break
        time.sleep(0.02)
    goto = [c for c in motion.mini.calls if isinstance(c, tuple)]
    assert goto == [("goto", 0.01, [0.1, -0.2], 0.3)]


def test_unknown_pose_lists_saved_ones(motion):
    motion.save_pose("bow")
    with pytest.raises(ValueError, match="Saved poses: bow"):
        motion.goto_pose("salute")


def test_lock_without_saving(motion):
    motion.learn_start()
    assert "locked" in motion.learn_stop()
    assert motion.mini.calls[-3:] == ["hold", "motors_on", "hold"] and not motion.learning


def test_empty_name_does_not_touch_motors(motion):
    motion.learn_start()
    with pytest.raises(ValueError):
        motion.save_pose("  ")
    assert motion.learning
