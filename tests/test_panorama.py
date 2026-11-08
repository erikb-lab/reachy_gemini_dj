"""reachy_gemini_dj/panorama.py: shot plan, pointing, alignment and stitching (synthetic data, no robot)."""
import json
import math

import numpy as np
import pytest

from reachy_gemini_dj import panorama as P

cv2 = pytest.importorskip("cv2")


def pose(roll=0.0, pitch=0.0, yaw=0.0, scale=1.0):
    T = np.eye(4)
    T[:3, :3] = P.rot_xyz(roll, pitch, yaw) * scale
    return T


# ------------------------------------------------------------------------------------------------ geometry ---
def test_head_ypr_round_trip():
    for y, p, r in [(0, 0, 0), (37, -20, 0), (-170, 35, 10), (120, 12.5, -22)]:
        got = P.head_ypr(pose(r, p, y))
        assert got == pytest.approx((y, p, r), abs=1e-9)


def test_camera_rotation_is_orthonormal_even_if_the_pose_is_not():
    # the daemon's pose matrices are ~0.05 % off a rotation; that made the bundle adjustment drift
    R = P.camera_rotation(pose(5, -12, 40, scale=1.0005))
    assert np.allclose(R.T @ R, np.eye(3), atol=1e-12) and np.linalg.det(R) == pytest.approx(1.0)


# ------------------------------------------------------------------------------------------------- the plan ---
@pytest.mark.parametrize("level,n", [("full", 42), ("quick", 25), ("ultra_fast", 15), ("fast", 25), ("regular", 42), ("slow", 60), ("ultra_slow", 80)])
def test_plan_reachable_and_serpentine(level, n):
    shots = P.plan(level)
    assert len(shots) == n
    for s in shots:
        assert -180 <= s.yaw <= 180 and abs(s.pitch) <= P.PITCH_LIMIT and abs(s.body) <= P.BODY_LIMIT
        assert abs(s.yaw - s.body) <= P.HEAD_TURN_LIMIT + 1e-9
        assert abs(s.roll) <= P.roll_limit(s.pitch, s.yaw - s.body) + 1e-9      # solvable by the robot's IK
    # never swings the body back across the whole range between neighbouring shots, except when changing row
    jumps = sum(abs(a.body - b.body) > 200 for a, b in zip(shots, shots[1:]))
    assert jumps == 0


def test_roll_limit_matches_the_ik_envelope():
    # from the robot's own inverse kinematics: roll shrinks towards the up-looking pitch limit and with head-body yaw
    assert P.roll_limit(-40) == 30 and P.roll_limit(-34) == 34 and P.roll_limit(40) == 37
    assert P.roll_limit(-40, 20) == 26 < P.roll_limit(-40, 0)
    assert P.roll_limit(-40, 90) == P.roll_limit(-40, 20)            # beyond HEAD_TURN_LIMIT: the worst case


@pytest.mark.parametrize("level,sphere", [("ultra_fast", 0.94), ("fast", 0.95), ("regular", 0.95), ("slow", 0.95), ("ultra_slow", 0.95)])
def test_plans_see_most_of_the_sphere(level, sphere):
    # the full-field-of-view camera (3072x1728), which is what the robot captures with
    cov = P.coverage(P.plan(level), P.Camera(size=(3072, 1728)), 360)
    assert cov["sphere"] > sphere
    assert cov["full_ring_up_to"] > 65 and cov["full_ring_down_to"] < -70


# ---------------------------------------------------------------------------------------------- pointing ---
class LaggyMini:
    """Fake robot whose head stops short in pitch and whose body stops short near the ends (as measured)."""

    def __init__(self):
        self.pose, self.moves = np.eye(4), []

    def goto_target(self, head, body_yaw, duration):
        y, p, r = P.head_ypr(head)
        b = math.degrees(body_yaw)
        self.moves.append((y, p, b))
        lag_y = 0.06 * b if abs(b) > 100 else 0.0
        self.pose = pose(r, p - 8 * np.sign(p), y - lag_y)

    def get_current_head_pose(self):
        return self.pose


def chp(roll, pitch, yaw, degrees=True):
    return pose(roll, pitch, yaw)


def job(tmp_path):
    return P.PanoramaJob(get_mini=lambda: None, store=tmp_path, motion_enabled=lambda: True)


def test_point_corrects_lag(tmp_path):
    m = LaggyMini()
    job(tmp_path)._point(m, P.Shot(60, -12.5), 0.5, chp)
    y, p, _ = P.head_ypr(m.get_current_head_pose())
    assert abs(p + 12.5) < P.POINT_TOL_DEG and abs(y - 60) < P.POINT_TOL_DEG
    assert len(m.moves) == 2


def test_point_stops_at_the_limit(tmp_path):
    m = LaggyMini()
    job(tmp_path)._point(m, P.Shot(-180, P.PITCH_LIMIT), 0.5, chp)
    assert len(m.moves) == 1                          # asking for more than the limit would not help
    assert all(abs(p) <= P.PITCH_LIMIT for _, p, _ in m.moves)


# --------------------------------------------------------------------------------------------- alignment ---
def synthetic_matches(cam, true_R, rng, per_pair=60):
    """Pixel matches between overlapping photos of random far-away points, from the true orientations."""
    pts = rng.normal(size=(6000, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    proj = []
    for R in true_R:
        c = pts @ R                                       # world -> camera
        ok = c[:, 2] > 0.3
        pix, _ = cv2.projectPoints(c[ok], np.zeros(3), np.zeros(3), cam.K, cam.D)
        pix = pix.reshape(-1, 2)
        ins = (pix[:, 0] > 5) & (pix[:, 0] < cam.w - 5) & (pix[:, 1] > 5) & (pix[:, 1] < cam.h - 5)
        idx = np.flatnonzero(ok)[ins]
        proj.append(dict(zip(idx.tolist(), pix[ins])))
    out = []
    for i in range(len(true_R)):
        for j in range(i + 1, len(true_R)):
            common = sorted(set(proj[i]) & set(proj[j]))[:per_pair]
            if len(common) >= 12:
                out.append((i, j, np.array([proj[i][k] for k in common]), np.array([proj[j][k] for k in common])))
    return out


def test_refine_recovers_pose_errors():
    rng = np.random.default_rng(1)
    cam = P.Camera()
    shots = [P.Shot(y, p) for p in (-25, 0, 25) for y in range(-180, 180, 40)]
    true_R = [P.camera_rotation(pose(0, s.pitch, s.yaw)) for s in shots]
    frames = []
    for s in shots:
        noisy = pose(rng.normal(0, 1.5), s.pitch + rng.normal(0, 1.5), s.yaw + rng.normal(0, 1.5))
        frames.append(P.Frame(None, noisy))
    before = max(math.degrees(math.acos(np.clip((np.trace(f.R @ t.T) - 1) / 2, -1, 1))) for f, t in zip(frames, true_R))
    info = P.refine(cam, frames, synthetic_matches(cam, true_R, rng), iters=20)
    assert info["median_deg"] < 0.05
    # relative orientations come out right to a fraction of a degree (the prior pulls each photo a little)
    rel = [f.R @ frames[0].R.T @ true_R[0] @ t.T for f, t in zip(frames, true_R)]
    err = max(math.degrees(math.acos(np.clip((np.trace(E) - 1) / 2, -1, 1))) for E in rel)
    assert err < 0.5 < before


# ---------------------------------------------------------------------------------------- stitch + files ---
def test_stitch_and_encode_small_synthetic_room():
    rng = np.random.default_rng(0)
    w, h = 640, 360
    cam = P.Camera(size=(w, h))
    world = cv2.resize(rng.integers(0, 255, (64, 128, 3), dtype=np.uint8), (2048, 1024), interpolation=cv2.INTER_CUBIC)
    frames = []
    for s in P.plan("quick"):
        R = P.camera_rotation(pose(s.roll, s.pitch, s.yaw))
        gx, gy = np.meshgrid(np.arange(w), np.arange(h))
        rays = cam.rays(np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64)) @ R.T
        lon = np.arctan2(rays[:, 1], rays[:, 0]); lat = np.arcsin(np.clip(rays[:, 2], -1, 1))
        mx = P.lon_to_x(lon, 2048).reshape(h, w).astype(np.float32) % 2048
        my = P.lat_to_y(lat, 1024).reshape(h, w).astype(np.float32)
        frames.append(P.Frame(cv2.remap(world, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP), pose(s.roll, s.pitch, s.yaw)))
    pano, info = P.stitch(frames, out_w=512)
    assert pano.shape == (256, 512, 3) and info["frames"] == len(frames)
    assert info["align"]["median_deg"] < 0.5
    full, thumb = P.encode(pano, len(frames))
    assert full[:2] == b"\xff\xd8" and b"GPano:ProjectionType" in full and b"equirectangular" in full
    assert cv2.imdecode(np.frombuffer(thumb, np.uint8), cv2.IMREAD_COLOR).shape[1] == 640


def test_job_files_list_path_delete(tmp_path):
    j = job(tmp_path)
    pid = "20261108-000049"
    (tmp_path / f"{pid}.jpg").write_bytes(b"\xff\xd8x")
    (tmp_path / f"{pid}_thumb.jpg").write_bytes(b"\xff\xd8t")
    (tmp_path / f"{pid}.json").write_text(json.dumps({"created": "x", "frames": 40}))
    (tmp_path / "last_raw").mkdir()
    assert [i["id"] for i in j.list()] == [pid]
    assert j.path(pid) and j.path(pid, thumb=True)
    for bad in ("../secret", "20261108-000049/../../x", "", None):
        assert j.path(bad) is None and j.delete(bad) is False
    assert j.delete(pid) and j.list() == []


def test_job_refuses_without_robot_or_movement(tmp_path):
    assert not job(tmp_path).start()["ok"]
    j = P.PanoramaJob(get_mini=lambda: object(), store=tmp_path, motion_enabled=lambda: False)
    assert "Movement" in j.start()["error"]


def test_pick_out_w_follows_photo_detail_and_memory():
    assert P.pick_out_w(1280, avail_mb=100) == P.OUT_W_STREAM            # the 720p stream: ~5600 px of detail
    assert P.pick_out_w(3072, avail_mb=8000) == 8192
    assert P.pick_out_w(3072, avail_mb=3100) == 6144                     # a robot with ~3 GB free
    assert P.pick_out_w(3072, avail_mb=900) == 4096                      # never refuse, just go smaller
    assert P.pick_out_w(3072, avail_mb=None) in P.OUT_W_STEPS


def test_fold_wrap_keeps_size_and_blends_the_seam():
    w, pad = 256, 32
    canvas = np.full((16, w + 2 * pad, 3), 100, np.uint8)
    mask = np.full((16, w + 2 * pad), 255, np.uint8)
    out, om = P.fold_wrap(canvas, mask, w, pad)
    assert out.shape == (16, w, 3) and out.dtype == np.uint8 and (out == 100).all() and (om == 255).all()


def test_delete_removes_the_raw_photos_of_that_capture(tmp_path):
    j = job(tmp_path)
    for pid in ("20261108-000049", "20261108-000150"):
        (tmp_path / f"{pid}.jpg").write_bytes(b"\xff\xd8x")
        (tmp_path / f"{pid}_thumb.jpg").write_bytes(b"\xff\xd8t")
        (tmp_path / f"{pid}.json").write_text(json.dumps({"created": "x", "frames": 40}))
    j._save_raw([P.Frame(np.zeros((8, 8, 3), np.uint8), np.eye(4))], "20261108-000150")
    assert (tmp_path / "last_raw" / "00.jpg").exists()
    assert j.delete("20261108-000049") and (tmp_path / "last_raw").exists()      # an older photo: keep the latest raw
    assert j.delete("20261108-000150") and not (tmp_path / "last_raw").exists()  # its own raw photos go with it
    assert list(tmp_path.iterdir()) == []                                        # nothing left behind


def test_modes_listing_start_and_unknown_mode(tmp_path):
    ls = P.modes_listing()
    assert [m["id"] for m in ls] == ["ultra_fast", "fast", "regular", "slow", "ultra_slow"]
    assert [m["shots"] for m in ls] == [15, 25, 42, 60, 80]
    assert [m["seconds"] for m in ls] == sorted(m["seconds"] for m in ls)
    with pytest.raises(ValueError):
        P.plan("bogus")
    r = job(tmp_path).start("page", "bogus")
    assert not r["ok"] and "Unknown mode" in r["error"]


def test_every_mode_is_reachable_and_overlaps_enough():
    cam = P.Camera(size=(3072, 1728))
    for mode in P.MODES:
        shots = P.plan(mode)
        assert all(abs(s.yaw - s.body) <= P.HEAD_TURN_LIMIT + 1e-9 and abs(s.pitch) <= P.PITCH_LIMIT for s in shots)
        assert all(abs(s.roll) <= P.roll_limit(s.pitch, s.yaw - s.body) + 1e-9 for s in shots)
        assert sum(abs(a.body - b.body) > 200 for a, b in zip(shots, shots[1:])) == 0


def test_pick_out_w_shrinks_for_many_photos():
    assert P.pick_out_w(3072, avail_mb=3000, n_frames=40) == 6144
    assert P.pick_out_w(3072, avail_mb=3000, n_frames=120) == 4096
