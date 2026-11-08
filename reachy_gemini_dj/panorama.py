"""360° photo: the robot turns its body and head through a grid of poses, takes a photo at each, and stitches them
into one equirectangular ("photo sphere") JPEG with Google's GPano metadata, so any 360 viewer shows it.

How it works
* Plan: rows of shots looking up, level and down. The BODY does the turning (±160°, head yaw only tops it up to
  ±180°), so the head stays centred where the Stewart platform can tilt furthest. Extra shots with the head rolled
  sideways while looking up / down push the long diagonal of the frame towards the zenith and nadir.
* Capture: after each move the robot waits until the head is still, then takes a fresh frame and records the head
  pose the motors actually reached (not the one that was asked for: they differ by 1-3°).
* Align: every photo starts at its measured orientation. Features matched between overlapping photos then refine
  all orientations (and the lens focal length) at once with a small robust bundle adjustment that is held close to
  the measured poses, so the horizon stays level and photos of blank walls/ceilings keep their measured place.
* Exposure: the camera's auto exposure and white balance change from photo to photo; per-photo, per-channel gains
  are solved from the overlaps so they match.
* Blend: OpenCV's graph-cut seam finder picks seams through low-detail areas, then a multi-band blender hides them.
  The 360° wrap is handled by stitching a slightly wider canvas and cross-fading the two ends.
* Poles: the robot can't look straight up or down. The small caps it cannot see are filled from their surroundings
  in a top-down (azimuthal) view, so there is no hole or smear line at the poles.

Everything here except ``capture()`` is pure numpy/OpenCV and is tested on the desktop with recorded frames.
"""
from __future__ import annotations

import io
import json
import logging
import math
import os
import queue
import shutil
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

# camera optical axes (x right, y down, z forward) expressed in the head frame (x forward, y left, z up)
CAM_IN_HEAD = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])

# Reachy Mini Wireless camera, 1280x720 stream. Self-calibrated from feature matches between photos taken at known
# head poses (the SDK's K is for the Lite camera's sensor and is ~30-60% off for this stream): square pixels,
# f = 892 px (70° x 44° field of view), mild radial distortion, and the camera sits ~2° down / 2.5° sideways in the head.
CALIB_SIZE = (1280, 720)
DEFAULT_K = np.array([[891.77, 0.0, 640.0], [0.0, 891.77, 360.0], [0.0, 0.0, 1.0]])
DEFAULT_D = np.array([0.0017, 0.0982, 0.0, 0.0, 0.0])
MOUNT_ROTVEC = np.array([-0.00285, 0.03769, -0.04313])     # radians, applied in the head frame
# The 1280x720 stream is the middle 3072x1728 of the 4608x2592 sensor, binned 2x2 to 1536x864 and scaled to 1280:
# so the focal length in full-resolution sensor pixels is 891.77 * 1536/1280 * 2. Other sizes derive from it.
F_NATIVE = 891.77 * 1536 / 1280 * 2          # 2140.2 px
SENSOR_SIZE = (4608, 2592)
# Lens models per stream size (K, D). Full-FOV sizes see out to the corners of the wide lens, which the 720p
# calibration never did, so they get their own calibration (CALIBRATIONS is filled in below).
CALIBRATIONS: dict = {}

BODY_LIMIT = 160.0            # official body yaw range, degrees
PITCH_LIMIT = 40.0            # official head pitch / roll range
OUT_W = 4096                  # fallback equirect output width; see pick_out_w
OUT_W_STREAM = 5632           # the 1280x720 stream carries ~5600 px of detail around the sphere
OUT_W_STEPS = (8192, 6144, 4096)   # full-resolution photos carry ~9000 px; the biggest one that fits in memory is used
# Measured peak memory of stitch() + encode at these widths on 40 photos (MB over what the app already holds):
# 6144 -> ~1150, 8192 -> ~1800. The blender's pyramids dominate and scale with the pixel count.
JPEG_QUALITY = 90
KEEP = 30                     # photos kept on the robot; the oldest are deleted beyond this


# ----------------------------------------------------------------------------------------------- geometry ---
def rot_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Rotation like reachy_mini.utils.create_head_pose (scipy 'xyz' extrinsic, degrees): Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (math.radians(v) for v in (roll, pitch, yaw))
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def expmap(w: np.ndarray) -> np.ndarray:
    th = float(np.linalg.norm(w))
    if th < 1e-12:
        return np.eye(3) + skew(w)
    k = skew(w / th)
    return np.eye(3) + math.sin(th) * k + (1 - math.cos(th)) * (k @ k)


def lonlat_to_dir(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cl = np.cos(lat)
    return cl * np.cos(lon), cl * np.sin(lon), np.sin(lat)


def x_to_lon(x, w: int):
    """Equirect column -> longitude (rad). The centre column looks forward; moving right turns right (clockwise)."""
    return math.pi - (np.asarray(x, dtype=np.float64) + 0.5) * (2 * math.pi / w)


def y_to_lat(y, h: int):
    return math.pi / 2 - (np.asarray(y, dtype=np.float64) + 0.5) * (math.pi / h)


def lon_to_x(lon, w: int):
    return (math.pi - np.asarray(lon)) * (w / (2 * math.pi)) - 0.5


def lat_to_y(lat, h: int):
    return (math.pi / 2 - np.asarray(lat)) * (h / math.pi) - 0.5


# ---------------------------------------------------------------------------------------------- the camera ---
class Camera:
    """Lens model: pixel <-> ray in camera coordinates, using OpenCV's (rational + thin prism) distortion."""

    def __init__(self, K=None, D=None, size=CALIB_SIZE):
        import cv2
        if K is None and tuple(size) in CALIBRATIONS:
            K, D0 = CALIBRATIONS[tuple(size)]
            D = D0 if D is None else D
        elif K is None and size[0] * SENSOR_SIZE[1] == size[1] * SENSOR_SIZE[0] and size[0] > 2000:
            # full field of view (libcamera's full-resolution sensor mode, scaled): focal length from F_NATIVE
            f = F_NATIVE * size[0] / SENSOR_SIZE[0]
            K = np.array([[f, 0.0, size[0] / 2], [0.0, f, size[1] / 2], [0.0, 0.0, 1.0]])
        elif K is None:                    # the 720p calibration, scaled if the stream is another size (same FOV)
            K = DEFAULT_K.copy()
            sx, sy = size[0] / CALIB_SIZE[0], size[1] / CALIB_SIZE[1]
            K[0] *= sx
            K[1] *= sy
        self.K = np.array(K, dtype=np.float64)
        self.D = np.array(DEFAULT_D if D is None else D, dtype=np.float64).ravel()
        self.w, self.h = int(size[0]), int(size[1])
        # the normalised-coordinate box every pixel maps into: rays outside it would fold back into the image
        n = 64
        border = np.concatenate([
            np.stack([np.linspace(0, self.w - 1, n), np.zeros(n)], 1), np.stack([np.linspace(0, self.w - 1, n), np.full(n, self.h - 1)], 1),
            np.stack([np.zeros(n), np.linspace(0, self.h - 1, n)], 1), np.stack([np.full(n, self.w - 1), np.linspace(0, self.h - 1, n)], 1)])
        nb = self.undistort(border)
        top, bot, left, right = nb[:n], nb[n:2 * n], nb[2 * n:3 * n], nb[3 * n:]
        self.box = (float(left[:, 0].max()), float(right[:, 0].min()), float(top[:, 1].max()), float(bot[:, 1].min()))
        # dense table normalised (x, y) -> pixel, so warping never calls projectPoints per output pixel
        x0, x1, y0, y1 = self.box
        self.lut_n = 768
        gx, gy = np.meshgrid(np.linspace(x0, x1, self.lut_n), np.linspace(y0, y1, self.lut_n))
        pts = np.stack([gx.ravel(), gy.ravel(), np.ones(gx.size)], 1)
        pix, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), self.K, self.D)
        pix = pix.reshape(self.lut_n, self.lut_n, 2).astype(np.float32)
        self.lut_x, self.lut_y = np.ascontiguousarray(pix[..., 0]), np.ascontiguousarray(pix[..., 1])
        self.half_fov = (math.degrees(math.atan(max(-x0, x1))), math.degrees(math.atan(max(-y0, y1))))

    def undistort(self, pix) -> np.ndarray:
        import cv2
        p = np.asarray(pix, dtype=np.float64).reshape(-1, 1, 2)
        crit = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 60, 1e-10)
        try:
            n = cv2.undistortPointsIter(p, self.K, self.D, None, None, crit)
        except AttributeError:
            n = cv2.undistortPoints(p, self.K, self.D, criteria=crit)   # OpenCV 5: default is only 5 iterations
        return n.reshape(-1, 2)

    def rays(self, pix, scale=(1.0, 1.0)) -> np.ndarray:
        """Unit rays (camera coordinates) for pixels; scale refines the focal length (normalised coords / scale)."""
        n = self.undistort(pix)
        v = np.stack([n[:, 0] / scale[0], n[:, 1] / scale[1], np.ones(len(n))], 1)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def normalised_to_pixel_maps(self, nx: np.ndarray, ny: np.ndarray):
        """Look up pixel maps (for cv2.remap) for normalised coordinates via the dense table."""
        import cv2
        x0, x1, y0, y1 = self.box
        fx = ((nx - x0) * ((self.lut_n - 1) / (x1 - x0))).astype(np.float32)
        fy = ((ny - y0) * ((self.lut_n - 1) / (y1 - y0))).astype(np.float32)
        mx = cv2.remap(self.lut_x, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e4)
        my = cv2.remap(self.lut_y, fx, fy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=-1e4)
        return mx, my

    def inside(self, nx, ny, margin=0.0):
        x0, x1, y0, y1 = self.box
        return (nx > x0 + margin) & (nx < x1 - margin) & (ny > y0 + margin) & (ny < y1 - margin)


def head_ypr(head_pose) -> tuple[float, float, float]:
    """(yaw, pitch, roll) in degrees of a 4x4 head pose, matching create_head_pose (extrinsic xyz)."""
    R = np.asarray(head_pose, dtype=np.float64)[:3, :3]
    return (math.degrees(math.atan2(R[1, 0], R[0, 0])), math.degrees(math.asin(max(-1.0, min(1.0, -R[2, 0])))),
            math.degrees(math.atan2(R[2, 1], R[2, 2])))


def camera_rotation(head_pose) -> np.ndarray:
    """World-from-camera rotation for a 4x4 head pose (the head pose is in the world frame, body yaw included).
    The daemon's pose matrix is not exactly orthonormal (det ~1.0005), which makes the bundle adjustment indefinite,
    so it is snapped to the nearest rotation first."""
    u, _, vt = np.linalg.svd(np.asarray(head_pose, dtype=np.float64)[:3, :3])
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        rot = u @ np.diag([1.0, 1.0, -1.0]) @ vt
    return rot @ expmap(MOUNT_ROTVEC) @ CAM_IN_HEAD


# --------------------------------------------------------------------------------------------- the shot plan ---
@dataclass
class Shot:
    yaw: float              # world yaw of the head, degrees (+ = left)
    pitch: float            # + = looking down
    roll: float = 0.0

    @property
    def body(self) -> float:
        return max(-BODY_LIMIT, min(BODY_LIMIT, self.yaw))


# Largest |roll| the head can take at a given pitch, from the robot's own inverse kinematics (the Rust solver in
# reachy_mini_rust_kinematics 1.0.3 with the shipped kinematics_data.json: inverse_kinematics_safe, max body yaw 160,
# max head-body yaw 65, forward kinematics read back), at head-body yaw 0 and 20 degrees (HEAD_TURN_LIMIT), worst of the
# two roll directions, minus 9 degrees: the measured roll overshoots the command by 3-6 degrees, and asking for the
# very edge made the head press against its limit (IK warnings, motor strain), so keep clear of it. Pitch and roll
# are not independent: at pitch -40 the solver stops at roll 33 (29 with the head turned 20 degrees from the body),
# at pitch 40 it reaches the full 40. (Pitch beyond +-40 is solvable too, but +-40 is the official limit.)
_ROLL_PITCH = [-40, -38, -36, -34, -32, -30, -20, 0, 40]
_ROLL_REL0 = [24, 25, 26, 28, 30, 31, 31, 31, 31]
_ROLL_REL20 = [20, 21, 23, 25, 26, 28, 31, 31, 30]


def roll_limit(pitch: float, rel_yaw: float = 0.0) -> float:
    """Reachable |roll| (degrees) at this pitch with the head turned rel_yaw degrees away from the body."""
    f = min(1.0, abs(rel_yaw) / HEAD_TURN_LIMIT)
    r0, r20 = np.interp(pitch, _ROLL_PITCH, _ROLL_REL0), np.interp(pitch, _ROLL_PITCH, _ROLL_REL20)
    return float(r0 + (r20 - r0) * f)


def _ring(n: int, pitch: float, offset: float = 0.0, roll: float = 0.0, alternate: bool = False) -> list[Shot]:
    """n shots evenly around the yaw circle; roll (signed) is clipped to what the head reaches at that pitch."""
    out = []
    for i in range(n):
        yaw = -180 + (i + offset) * 360 / n
        rel = yaw - max(-BODY_LIMIT, min(BODY_LIMIT, yaw))
        r = math.copysign(min(abs(roll), roll_limit(pitch, rel)), roll)
        out.append(Shot(yaw, pitch, (r if i % 2 else -r) if alternate else r))
    return out


# Capture modes: rows of (shots, pitch, offset, roll) from the zenith row to the nadir row. More photos mean smaller
# steps where near things shift between neighbouring photos (the camera is 4-5 cm off the head's rotation centre) and
# better overlap for alignment, at about 2 s of capture and 3.5 s of stitching per photo. Coverage (share of the
# sphere seen at least once / at least twice), measured with the full-field-of-view camera:
#   ultra_fast 15: 93 % / 50 %   fast 25: 93 % / 85 %   regular 42: 93 % / 91 %   slow 60 and ultra_slow 80: 93 % / 92 %
# (the top and bottom rows keep clear of the head's limits: pitch 36 and roll 24 / 30, a little less than it can do)
_UP, _DOWN = -36.0, 36.0           # the rolled rows stay 4 degrees inside the pitch limit, away from its edge
MODES: dict[str, dict] = {
    "ultra_fast": {"label": "Ultra fast", "note": "preview: little overlap, steps and soft seams likely",
                   "rows": [(5, _UP, .5, -24), (5, 0, .25), (5, _DOWN, .25, 30)]},
    "fast": {"label": "Fast", "note": "good for most rooms",
             "rows": [(6, _UP, .5, -24), (6, -12, .25), (6, 14, .75), (7, _DOWN, .25, 30)]},
    "regular": {"label": "Regular", "note": "the default",
                "rows": [(9, _UP, .5, -24), (12, -12, .25), (12, 14, .75), (9, _DOWN, .25, 30)]},
    "slow": {"label": "Slow", "note": "smaller steps on near objects",
             "rows": [(10, _UP, .5, -24), (12, -26, .25), (14, -4, .75), (14, 18, .25), (10, _DOWN, .5, 30)]},
    "ultra_slow": {"label": "Ultra slow", "note": "smallest steps; the stitch may use a smaller picture size on the robot",
                   "rows": [(12, _UP, .5, -24), (14, -30, .25), (14, -14, .75), (14, 2, .25), (14, 18, .75), (12, _DOWN, .25, 30)]},
}
DEFAULT_MODE = "regular"
MODE_ALIASES = {"full": "regular", "quick": "fast", "normal": "regular", "default": "regular", "standard": "regular",
                "fastest": "ultra_fast", "ultrafast": "ultra_fast", "slowest": "ultra_slow", "ultraslow": "ultra_slow",
                "best": "ultra_slow", "preview": "ultra_fast"}


def normalise_mode(mode) -> str | None:
    """'Ultra slow', 'ultra-slow', ' FAST ', 'quick' ... -> a key of MODES, or None when it is not a mode."""
    if mode is None or str(mode).strip() == "":
        return DEFAULT_MODE
    key = str(mode).strip().lower().replace(" ", "_").replace("-", "_")
    key = MODE_ALIASES.get(key, key)
    return key if key in MODES else None


def mode_shots(mode: str) -> int:
    return sum(r[0] for r in MODES[MODE_ALIASES.get(mode, mode)]["rows"])


def mode_estimate_s(mode: str) -> int:
    """Rough total seconds on the robot (capture ~2.6 s and stitch ~3.6 s per photo, plus start and finish)."""
    n = mode_shots(mode)
    stitch_per = 3.6 if n <= 60 else 2.3          # 80 photos: the robot stitches at 4096 wide to stay within memory
    return int(round(2.6 * n + stitch_per * n + 30, -1))


def modes_listing() -> list[dict]:
    return [{"id": k, "label": v["label"], "shots": mode_shots(k), "seconds": mode_estimate_s(k), "note": v["note"]}
            for k, v in MODES.items()]


def plan(level: str = DEFAULT_MODE) -> list[Shot]:
    """Shots in serpentine order (the body never swings back across the whole range mid-capture).

    The camera sees ~94 x 62 degrees (87 x 56 inside the usable lens circle), so rows ~25 degrees apart overlap by half.
    Pitch is limited to +-40 and the camera rolls about its own axis, so looking up/down the only way to see further
    towards a pole is to roll the head: the top and bottom rows use the biggest roll the inverse kinematics solves
    (roll_limit), since every shot there reaches a bit further on one side. All modes see ~93 % of the sphere by area
    (everything from ~64 degrees above to ~70 below the horizon, a polar hole fill_poles completes). `level` is a key of
    MODES ("full" and "quick" still work as "regular" and "fast")."""
    given, level = level, normalise_mode(level)
    if level is None:
        raise ValueError(f"unknown 360 photo mode: {given!r}")
    rows = [_ring(*r) for r in MODES[level]["rows"]]
    out: list[Shot] = []
    for k, row in enumerate(rows):
        row = sorted(row, key=lambda s: s.yaw, reverse=bool(k % 2))
        out.extend(row)
    return out


# --------------------------------------------------------------------------------------------- alignment ---
@dataclass
class Frame:
    image: np.ndarray | None            # BGR, as captured (None once compressed to jpeg to save memory)
    pose: np.ndarray                    # 4x4 measured head pose
    R: np.ndarray = None                # world-from-camera, refined
    jpeg: bytes | None = None           # the photo, compressed (full-resolution captures keep only this)
    size: tuple = None                  # (width, height)

    def __post_init__(self):
        if self.R is None:
            self.R = camera_rotation(self.pose)
        if self.size is None and self.image is not None:
            self.size = (self.image.shape[1], self.image.shape[0])

    def load(self) -> np.ndarray:
        """The photo as BGR (decoded from jpeg if it was compressed)."""
        if self.image is not None:
            return self.image
        import cv2
        return cv2.imdecode(np.frombuffer(self.jpeg, np.uint8), cv2.IMREAD_COLOR)

    def compress(self, quality: int = 94) -> None:
        import cv2
        if self.image is not None:
            ok, buf = cv2.imencode(".jpg", self.image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            if ok:
                self.jpeg, self.image = buf.tobytes(), None


_CLAHE = None


def _feature(image: np.ndarray, scale: float = 0.5):
    """SIFT keypoints (full-resolution pixels) and descriptors of one photo."""
    import cv2
    global _CLAHE
    if _CLAHE is None:
        _CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    kp, des = cv2.SIFT_create(nfeatures=900, contrastThreshold=0.02).detectAndCompute(_CLAHE.apply(g), None)
    if des is None or not kp:
        return np.zeros((0, 2)), None
    return np.array([k.pt for k in kp], dtype=np.float64) / scale, des


def _overlap(Ri: np.ndarray, Rj: np.ndarray, max_angle_deg: float = 85.0) -> bool:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(Ri[:, 2] @ Rj[:, 2]))))) < max_angle_deg


class Matcher:
    """Feature matches between overlapping photos as (i, j, pixels_i, pixels_j), gated by the measured poses.
    Photos are added one at a time, so the capture can match each new photo while the robot moves to the next one
    (this is most of the stitching work; doing it during the capture hides it)."""

    def __init__(self, cam: Camera, gate_deg: float = 10.0):
        import cv2
        self.cam, self.cos_gate = cam, math.cos(math.radians(gate_deg))
        self.bf = cv2.BFMatcher(cv2.NORM_L2)
        self.feats: list = []
        self.rots: list[np.ndarray] = []
        self.matches: list = []

    def add(self, frame: Frame) -> None:
        j = len(self.feats)
        self.feats.append(_feature(frame.load()))
        self.rots.append(frame.R.copy())
        for i in range(j):
            if _overlap(self.rots[i], self.rots[j]):
                m = self._pair(i, j)
                if m is not None:
                    self.matches.append(m)

    def _pair(self, i: int, j: int):
        (pi, di), (pj, dj) = self.feats[i], self.feats[j]
        if di is None or dj is None or len(pi) < 8 or len(pj) < 8:
            return None
        knn = self.bf.knnMatch(di, dj, k=2)
        good = [m[0] for m in knn if len(m) == 2 and m[0].distance < 0.8 * m[1].distance]
        if len(good) < 8:
            return None
        a = pi[[m.queryIdx for m in good]]
        b = pj[[m.trainIdx for m in good]]
        wa = self.cam.rays(a) @ self.rots[i].T
        wb = self.cam.rays(b) @ self.rots[j].T
        keep = np.einsum("ij,ij->i", wa, wb) > self.cos_gate
        if keep.sum() < 8:
            return None
        a, b, wa, wb = a[keep], b[keep], wa[keep], wb[keep]
        # one relative rotation must explain the pair: RANSAC on world rays (Kabsch on 2-point samples)
        best = None
        rng = np.random.default_rng(i * 1000 + j)
        for _ in range(60):
            smp = rng.choice(len(wa), 2, replace=False)
            U, _, Vt = np.linalg.svd(wb[smp].T @ wa[smp])
            Rr = U @ np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))]) @ Vt
            inl = np.linalg.norm(wa @ Rr.T - wb, axis=1) < math.radians(0.6)
            if best is None or inl.sum() > best.sum():
                best = inl
        if best is None or best.sum() < 10:
            return None
        return i, j, a[best], b[best]


def match_frames(cam: Camera, frames: list[Frame], gate_deg: float = 10.0):
    """All matches for a finished set of photos (see Matcher)."""
    m = Matcher(cam, gate_deg)
    for f in frames:
        m.add(f)
    return m.matches


def refine(cam: Camera, frames: list[Frame], matches, prior_deg: float = 3.0, match_deg: float = 0.3,
           iters: int = 8) -> dict:
    """Bundle adjustment of all photo orientations. Each photo gets a small world-frame correction w_i
    (R_i <- exp(w_i) R_i0). Matched rays must agree (noise ~match_deg); each correction is held near zero with
    ~prior_deg of slack (the measured poses are good to a few degrees), which fixes the gauge, keeps the horizon level
    and leaves photos without matches (blank ceiling) at their measured pose. Gauss-Newton, outliers dropped as it
    converges. The lens is calibrated (see DEFAULT_K), so only rotations are solved: fast and stable."""
    n = len(frames)
    if not matches:
        return {"matches": 0, "pairs": 0, "rms_deg": None}
    R0 = [f.R.copy() for f in frames]
    obs = [(i, j, cam.rays(a), cam.rays(b)) for i, j, a, b in matches]
    keep = [np.ones(len(ra), bool) for _, _, ra, _ in obs]
    w = np.zeros((n, 3))
    wd = 1.0 / math.radians(match_deg) ** 2
    wp = 1.0 / math.radians(prior_deg) ** 2

    def world(k_rot, i):
        return expmap(w[i]) @ R0[i]

    for it in range(iters):
        Rs = [expmap(w[k]) @ R0[k] for k in range(n)]
        A = np.eye(3 * n) * wp
        g = wp * w.ravel().copy()
        all_res = []
        for (i, j, ra, rb), kp in zip(obs, keep):
            va, vb = ra[kp] @ Rs[i].T, rb[kp] @ Rs[j].T
            r = va - vb                                       # (m, 3)
            all_res.append(np.linalg.norm(r, axis=1))
            # d(exp(w) v)/dw = -[v]x  ->  J_i = -[va]x, J_j = +[vb]x ; sum over matches of J^T J and J^T r
            Sa = np.einsum("mi,mj->ij", va, va); Sb = np.einsum("mi,mj->ij", vb, vb)
            Sab = np.einsum("mi,mj->ij", va, vb)
            m = len(va)
            # [v]x^T [v]x = |v|^2 I - v v^T (|v| = 1)
            A[3*i:3*i+3, 3*i:3*i+3] += wd * (m * np.eye(3) - Sa)
            A[3*j:3*j+3, 3*j:3*j+3] += wd * (m * np.eye(3) - Sb)
            # cross term: (-[va]x)^T [vb]x = [va]x [vb]x = vb va^T - (va.vb) I
            cross = np.einsum("mi,mj->ij", vb, va) - np.einsum("mi,mi->", va, vb) * np.eye(3)
            A[3*i:3*i+3, 3*j:3*j+3] += wd * cross
            A[3*j:3*j+3, 3*i:3*i+3] += wd * cross.T
            # J_i^T r = [va]x r = va x r ; J_j^T r = -[vb]x r = -(vb x r)
            g[3*i:3*i+3] += wd * np.cross(va, r).sum(0)
            g[3*j:3*j+3] -= wd * np.cross(vb, r).sum(0)
        step = np.linalg.solve(A, -g).reshape(n, 3)
        w = w + step
        # drop outliers once the solution has settled a bit
        if it >= 1:
            Rs = [expmap(w[k]) @ R0[k] for k in range(n)]
            res = np.concatenate([np.linalg.norm(ra @ Rs[i].T - rb @ Rs[j].T, axis=1) for i, j, ra, rb in obs])
            thr = max(math.radians(1.0), 4 * float(np.median(res)))
            ofs = 0
            for k, (i, j, ra, rb) in enumerate(obs):
                keep[k] = res[ofs:ofs + len(ra)] < thr
                ofs += len(ra)
        if np.abs(step).max() < 1e-6:
            break
    for k in range(n):
        frames[k].R = expmap(w[k]) @ R0[k]
    res = np.concatenate([np.linalg.norm(ra[kp] @ frames[i].R.T - rb[kp] @ frames[j].R.T, axis=1)
                          for (i, j, ra, rb), kp in zip(obs, keep)])
    return {"matches": int(len(res)), "pairs": len(matches), "rms_deg": float(np.degrees(np.sqrt(np.mean(res ** 2)))),
            "median_deg": float(np.degrees(np.median(res))),
            "max_correction_deg": float(np.degrees(np.linalg.norm(w, axis=1).max()))}


# ----------------------------------------------------------------------------------------------- warping ---
@dataclass
class Warp:
    image: np.ndarray        # BGR uint8, the frame's part of the canvas
    mask: np.ndarray         # uint8 0/255
    corner: tuple            # (x, y) top-left on the canvas
    weight: np.ndarray = field(default=None, repr=False)


def _frame_lon_lat_range(cam: Camera, R: np.ndarray, w: int, h: int):
    """Equirect bbox (x0, x1, y0, y1) of a frame, x unwrapped around the frame's centre (may run past 0 / w)."""
    x0b, x1b, y0b, y1b = cam.box
    t = np.linspace(0, 1, 40)
    nx = np.concatenate([x0b + (x1b - x0b) * t, np.full(40, x1b), x0b + (x1b - x0b) * t, np.full(40, x0b)])
    ny = np.concatenate([np.full(40, y0b), y0b + (y1b - y0b) * t, np.full(40, y1b), y0b + (y1b - y0b) * t])
    v = np.stack([nx, ny, np.ones_like(nx)], 1) @ R.T
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    lon, lat = np.arctan2(v[:, 1], v[:, 0]), np.arcsin(np.clip(v[:, 2], -1, 1))
    c = R[:, 2]
    lon_c = math.atan2(c[1], c[0])
    pole = None
    for z in (1.0, -1.0):                               # does the frame contain a pole?
        pc = R.T @ np.array([0, 0, z])
        if pc[2] > 0 and cam.inside(pc[0] / pc[2], pc[1] / pc[2]):
            pole = z
    if pole is not None:
        ys = lat_to_y(lat, h)
        return 0.0, float(w), (0.0 if pole > 0 else float(ys.min())), (float(h) if pole < 0 else float(ys.max())), True
    dl = (lon - lon_c + math.pi) % (2 * math.pi) - math.pi
    xs = lon_to_x(lon_c + dl, w)
    ys = lat_to_y(lat, h)
    return float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()), False


def warp_frame(cam: Camera, image: np.ndarray, R: np.ndarray, w: int, h: int, pad: int, scale=(1.0, 1.0),
               gains=None) -> list[Warp]:
    """Project one photo onto the equirect canvas [-pad, w + pad) x [0, h). Returns 1-2 pieces (a copy shifted by
    ±w when the photo crosses the canvas edge)."""
    import cv2
    x0, x1, y0, y1, full = _frame_lon_lat_range(cam, R, w, h)
    y0, y1 = max(0, int(math.floor(y0)) - 2), min(h, int(math.ceil(y1)) + 3)
    if full:
        spans = [(-pad, w + pad)]
    else:
        a, b = int(math.floor(x0)) - 2, int(math.ceil(x1)) + 3
        spans = []
        for shift in (-w, 0, w):
            s0, s1 = max(a + shift, -pad), min(b + shift, w + pad)
            if s1 > s0:
                spans.append((s0, s1))
    img = image
    if gains is not None:
        img = np.clip(image.astype(np.float32) * np.asarray(gains, np.float32)[None, None, :], 0, 255).astype(np.uint8)
    out = []
    Rt = R.T.astype(np.float32)
    for s0, s1 in spans:
        xs = np.arange(s0, s1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        lon = x_to_lon(xs, w).astype(np.float32)
        lat = y_to_lat(ys, h).astype(np.float32)
        cl, sl = np.cos(lat)[:, None], np.sin(lat)[:, None]
        dx, dy, dz = cl * np.cos(lon)[None, :], cl * np.sin(lon)[None, :], np.broadcast_to(sl, (len(ys), len(xs)))
        cx = Rt[0, 0] * dx + Rt[0, 1] * dy + Rt[0, 2] * dz
        cy = Rt[1, 0] * dx + Rt[1, 1] * dy + Rt[1, 2] * dz
        cz = Rt[2, 0] * dx + Rt[2, 1] * dy + Rt[2, 2] * dz
        front = cz > 1e-3
        czs = np.where(front, cz, 1.0)
        nx, ny = cx / czs * scale[0], cy / czs * scale[1]
        valid = front & cam.inside(nx, ny, 1e-3)
        if not valid.any():
            continue
        mx, my = cam.normalised_to_pixel_maps(np.where(valid, nx, -10), np.where(valid, ny, -10))
        valid &= (mx >= 0) & (mx <= cam.w - 1) & (my >= 0) & (my <= cam.h - 1)
        rows, cols = np.where(valid)
        if len(rows) == 0:
            continue
        r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
        mx, my, valid = mx[r0:r1, c0:c1], my[r0:r1, c0:c1], valid[r0:r1, c0:c1]
        piece = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mask = valid.astype(np.uint8) * 255
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8))
        # feather weight: distance from the photo's edge in the source image (vignetting-safe blending weight)
        out.append(Warp(piece, mask, (int(s0 + c0), int(y0 + r0))))
    return out


# --------------------------------------------------------------------------------------- exposure matching ---
def solve_gains(cam: Camera, frames: list[Frame], w: int = 512, sigma_n: float = 10.0, sigma_g: float = 0.1):
    """Per-photo, per-channel gains (OpenCV GainCompensator's least squares, done per channel so it also evens out
    white balance) from mean colours in pairwise overlaps on a small equirect."""
    h = w // 2
    small = [warp_frame(cam, cv2_resize_half(f.load()), f.R, w, h, 0) for f in frames]
    n = len(frames)
    # rasterise each photo's pieces on the small canvas
    imgs = np.zeros((n, h, w, 3), np.float32)
    masks = np.zeros((n, h, w), bool)
    for k, pieces in enumerate(small):
        for p in pieces:
            x, y = p.corner
            ph, pw = p.mask.shape
            sl = (slice(y, y + ph), slice(max(x, 0), min(x + pw, w)))
            cx0 = max(x, 0) - x
            cx1 = cx0 + (sl[1].stop - sl[1].start)
            if cx1 <= cx0:
                continue
            m = p.mask[:, cx0:cx1] > 0
            imgs[k][sl][m] = p.image[:, cx0:cx1][m]
            masks[k][sl] |= m
    gains = np.ones((n, 3))
    for ch in range(3):
        A = np.zeros((n, n))
        b = np.zeros(n)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ov = masks[i] & masks[j]
                N = int(ov.sum())
                if N < 50:
                    continue
                Ii = float(imgs[i][..., ch][ov].mean())
                Ij = float(imgs[j][..., ch][ov].mean())
                A[i, i] += N * (2 * Ii * Ii / sigma_n ** 2 + 1 / sigma_g ** 2)
                A[i, j] -= N * 2 * Ii * Ij / sigma_n ** 2
                b[i] += N / sigma_g ** 2
        for i in range(n):
            if A[i, i] == 0:
                A[i, i], b[i] = 1.0, 1.0
        gains[:, ch] = np.linalg.solve(A, b)
    return np.clip(gains, 0.4, 2.5)


def cv2_resize_half(img):
    import cv2
    return cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------------------------- blending ---
def blend(pieces: list[Warp], w: int, h: int, pad: int, seam_scale: float = 0.25, bands: int = 5):
    """Graph-cut seams on a small copy, then multi-band blend. Returns (canvas BGR uint8 [h, w+2pad], mask)."""
    import cv2
    # seams on a reduced copy
    sm_imgs, sm_masks, sm_corners = [], [], []
    for p in pieces:
        si = cv2.resize(p.image, None, fx=seam_scale, fy=seam_scale, interpolation=cv2.INTER_AREA)
        sm = cv2.resize(p.mask, (si.shape[1], si.shape[0]), interpolation=cv2.INTER_NEAREST)
        sm_imgs.append(si.astype(np.float32))
        sm_masks.append(cv2.UMat(sm))
        sm_corners.append((int(round((p.corner[0] + pad) * seam_scale)), int(round(p.corner[1] * seam_scale))))
    try:
        finder = cv2.detail_GraphCutSeamFinder("COST_COLOR_GRAD")
        sm_masks = finder.find(sm_imgs, sm_corners, sm_masks)
    except Exception as e:                       # very unlikely; fall back to plain overlaps (multi-band still hides)
        log.warning("seam finder failed (%s); blending without seams", e)
    feed = seam_masks_full(pieces, sm_masks, sm_corners, seam_scale, w + 2 * pad, h, pad)
    blender = cv2.detail_MultiBandBlender(0, bands)
    blender.prepare((0, 0, w + 2 * pad, h))
    for k, (p, m) in enumerate(zip(pieces, feed)):
        blender.feed(p.image.astype(np.int16), m, (p.corner[0] + pad, p.corner[1]))
        feed[k] = pieces[k].image = pieces[k].mask = None      # peak memory: the pieces are not needed after feeding
    res, res_mask = blender.blend(None, None)
    del blender
    np.clip(res, 0, 255, out=res)
    return res.astype(np.uint8), res_mask



def seam_masks_full(pieces: list[Warp], sm_masks, sm_corners, seam_scale: float, cw: int, ch: int, pad: int):
    """Full-resolution blend masks from the low-resolution graph-cut seams, with no gaps between neighbours.

    The seams are found on a 1/4 copy whose piece corners were rounded to whole low-res pixels (up to +-2 full-res
    pixels off) and whose sizes are not exactly 1/4 (resize rounding), so upscaling each piece's seam mask to its own
    rectangle put neighbouring pieces' seam boundaries a few pixels apart. Pixels that no piece claimed got blend
    weight 0, and MultiBandBlender paints weight-0 pixels black: the thin dark lines along seams. Here every piece
    samples the low-res seam masks on the SAME canvas grid (so shared boundaries coincide), and any pixel some photo
    sees but no seam mask claims is given to every photo that sees it."""
    import cv2
    s = seam_scale
    out = []
    claimed = np.zeros((ch, cw), np.uint8)
    for p, sm in zip(pieces, sm_masks):
        sm = sm.get() if isinstance(sm, cv2.UMat) else sm
        X0, Y0 = p.corner[0] + pad, p.corner[1]
        ph, pw = p.mask.shape
        lx, ly = sm_corners[len(out)]
        # full-res canvas pixel X -> low-res canvas coord (X + .5) * s - .5 -> piece-local low-res coord minus lx
        M = np.array([[s, 0, (X0 + 0.5) * s - 0.5 - lx], [0, s, (Y0 + 0.5) * s - 0.5 - ly]], np.float64)
        seam = cv2.warpAffine(sm, M, (pw, ph), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REPLICATE)
        seam = cv2.dilate((seam > 127).astype(np.uint8) * 255, np.ones((3, 3), np.uint8))
        m = cv2.bitwise_and(seam, p.mask)
        claimed[Y0:Y0 + ph, X0:X0 + pw] |= m
        out.append(m)
    for p, m in zip(pieces, out):           # pixels a photo sees that no seam mask claimed (seam ends, mask edges)
        X0, Y0 = p.corner[0] + pad, p.corner[1]
        ph, pw = p.mask.shape
        orphan = cv2.bitwise_and(p.mask, cv2.bitwise_not(claimed[Y0:Y0 + ph, X0:X0 + pw]))
        cv2.bitwise_or(m, orphan, dst=m)
    return out


def fold_wrap(canvas: np.ndarray, mask: np.ndarray, w: int, pad: int, k: int = 64):
    """Cut the [-pad, w+pad) canvas to [0, w) and cross-fade the two renderings of the 360° seam. Only the strips
    around the seam go through float (the 8192-wide canvas would need 400 MB as float)."""
    k = min(k, pad)
    out = np.ascontiguousarray(canvas[:, pad:pad + w])
    om = np.ascontiguousarray(mask[:, pad:pad + w])
    A = canvas[:, pad - k:pad + k].astype(np.float32)               # longitudes around the seam, left side
    B = canvas[:, pad + w - k:pad + w + k].astype(np.float32)       # the same longitudes, right side
    ma = mask[:, pad - k:pad + k] > 0
    mb = mask[:, pad + w - k:pad + w + k] > 0
    t = np.linspace(0, 1, 2 * k, dtype=np.float32)[None, :, None]
    wa = t * ma[..., None]
    wb = (1 - t) * mb[..., None]
    M = np.clip((A * wa + B * wb) / np.maximum(wa + wb, 1e-6) + 0.5, 0, 255).astype(np.uint8)   # rounded, not truncated
    mm = ma | mb
    out[:, w - k:] = M[:, :k]
    out[:, :k] = M[:, k:]
    om[:, w - k:] = mm[:, :k] * 255
    om[:, :k] = mm[:, k:] * 255
    return out, om


# ------------------------------------------------------------------------------------------- pole filling ---
def _pull_push(img: np.ndarray, known: np.ndarray) -> np.ndarray:
    """Fill unknown pixels smoothly from known ones (masked image pyramid down, then up); known pixels are kept."""
    import cv2
    levels = []
    cur, wgt = img.astype(np.float32) * known[..., None], known.astype(np.float32)
    while min(cur.shape[:2]) > 4:
        levels.append((cur, wgt))
        cur = cv2.resize(cur, (cur.shape[1] // 2 or 1, cur.shape[0] // 2 or 1), interpolation=cv2.INTER_AREA) * 4
        wgt = cv2.resize(wgt, (wgt.shape[1] // 2 or 1, wgt.shape[0] // 2 or 1), interpolation=cv2.INTER_AREA) * 4
        cur = cv2.GaussianBlur(cur, (3, 3), 0)
        wgt = cv2.GaussianBlur(wgt, (3, 3), 0)
    fill = cur / np.maximum(wgt, 1e-6)[..., None]
    for c, wg in reversed(levels):
        up = cv2.resize(fill, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
        up = cv2.GaussianBlur(up, (5, 5), 0)
        a = np.clip(wg, 0, 1)[..., None]
        fill = (c / np.maximum(wg, 1e-6)[..., None]) * a + up * (1 - a)
    return fill


def _add_texture(polar: np.ndarray, pk: np.ndarray, inside: np.ndarray, hole: np.ndarray, smooth: np.ndarray) -> np.ndarray:
    """Give the smooth pole fill some structure: mirror the surroundings across the edge of the hole (along each ray
    from the pole), keep only the mirror's detail (it minus its own blur, so no colour shift or seam) and fade it out
    towards the middle of the hole, where there is nothing to continue. Lines of the ceiling or floor run on into the
    hole and fold back, instead of ending in a flat blur."""
    import cv2
    size = polar.shape[0]
    R = size // 2
    nth = 720
    th = np.linspace(-math.pi, math.pi, nth, endpoint=False).astype(np.float32)
    rr = np.arange(R + 1, dtype=np.float32)
    ru = (R + np.cos(th)[:, None] * rr[None]).astype(np.float32)
    rv = (R + np.sin(th)[:, None] * rr[None]).astype(np.float32)
    hs = cv2.remap(hole.astype(np.uint8) * 255, ru, rv, cv2.INTER_NEAREST) > 127
    rb = np.where(hs.any(axis=1), R - np.argmax(hs[:, ::-1], axis=1), 0).astype(np.float32)   # last hole pixel per ray
    k = 9
    rb = np.convolve(np.concatenate([rb[-k:], rb, rb[:k]]), np.ones(2 * k + 1) / (2 * k + 1), mode="valid").astype(np.float32)
    if rb.max() < 4:
        return smooth
    u, v = np.meshgrid(np.arange(size) - R, np.arange(size) - R)
    rho = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)
    ang = np.arctan2(v, u).astype(np.float32)
    rbp = np.interp(ang, th, rb, period=2 * math.pi).astype(np.float32)
    src = 2 * rbp - rho
    su, sv = (R + np.cos(ang) * src).astype(np.float32), (R + np.sin(ang) * src).astype(np.float32)
    ok = hole & (src < R - 1) & (cv2.remap(pk.astype(np.uint8) * 255, su, sv, cv2.INTER_NEAREST, borderValue=0) > 127)
    mir = cv2.remap(np.clip(polar, 0, 255).astype(np.float32), su, sv, cv2.INTER_LINEAR)
    mir = np.where(ok[..., None], mir, smooth)
    sigma = max(3.0, float(rb[rb > 0].mean()) / 6)
    detail = mir - cv2.GaussianBlur(mir, (0, 0), sigma)
    w = np.clip(1.0 - (rbp - rho) / np.maximum(0.9 * rbp, 1.0), 0.0, 1.0) ** 0.7
    w = cv2.GaussianBlur(w * ok, (0, 0), 2)
    return smooth + detail * w[..., None]


def fill_poles(pano: np.ndarray, mask: np.ndarray, cap_deg: float = 60.0) -> tuple[np.ndarray, dict]:
    """Fill what the robot could not see around the zenith and nadir, working in a top-down view of each cap so
    the fill is round and smooth instead of smeared along the equirect rows."""
    import cv2
    h, w = mask.shape
    out = pano.copy()
    info = {}
    known_all = mask > 0
    for name, sign in (("zenith", 1), ("nadir", -1)):
        R = int(round(cap_deg / 180 * h))                          # cap radius in polar-view pixels (equirect scale)
        size = 2 * R + 1
        u, v = np.meshgrid(np.arange(size) - R, np.arange(size) - R)
        rho = np.sqrt(u.astype(np.float32) ** 2 + v.astype(np.float32) ** 2)           # pixels from the pole
        theta = np.arctan2(v, u).astype(np.float32)
        lat = sign * (math.pi / 2 - rho * (math.pi / h))
        lon = theta * sign
        mx = lon_to_x(lon, w).astype(np.float32) % w
        my = lat_to_y(lat, h).astype(np.float32)
        inside = rho <= R
        polar = cv2.remap(pano, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
        pk = cv2.remap(known_all.astype(np.uint8) * 255, mx, my, cv2.INTER_NEAREST, borderMode=cv2.BORDER_WRAP) > 127
        pk &= inside
        hole = inside & ~pk
        info[name] = {"missing_deg": float(np.degrees(np.sqrt(hole.sum() / math.pi) * math.pi / h)) if hole.any() else 0.0}
        if not hole.any():
            continue
        filled = _add_texture(polar, pk, inside, hole, _pull_push(polar, pk | ~inside))
        # map back: every equirect pixel in this cap that is a hole
        ys = np.arange(h)
        cap_rows = ys[(sign * y_to_lat(ys, h)) > math.radians(90 - cap_deg)]
        if len(cap_rows) == 0:
            continue
        y0, y1 = int(cap_rows.min()), int(cap_rows.max()) + 1
        lat_e = y_to_lat(np.arange(y0, y1), h)[:, None]
        lon_e = x_to_lon(np.arange(w), w)[None, :]
        rho_e = (math.pi / 2 - sign * lat_e) * (h / math.pi)
        th_e = lon_e * sign
        pu = (rho_e * np.cos(th_e) + R).astype(np.float32)
        pv = (rho_e * np.sin(th_e) + R).astype(np.float32)
        back = cv2.remap(np.clip(filled, 0, 255).astype(np.uint8), pu, pv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        holes_e = ~known_all[y0:y1]
        # soften the boundary a few pixels into the known area so the fill never shows a hard edge
        soft = cv2.GaussianBlur(holes_e.astype(np.float32), (0, 0), 3)
        soft = np.maximum(soft, holes_e.astype(np.float32))[..., None]
        out[y0:y1] = (back * soft + out[y0:y1] * (1 - soft)).astype(np.uint8)
    return out, info



def fill_gaps(pano: np.ndarray, mask: np.ndarray, cap_deg: float = 60.0, margin: int = 32) -> tuple[np.ndarray, int]:
    """Fill uncovered pixels outside the pole caps (fill_poles only works within cap_deg of the poles), e.g. a wedge
    between two rows of photos that did not overlap. Smooth pull-push fill from the surroundings, wrap-aware, with a
    soft edge; returns (pano, number of pixels filled)."""
    import cv2
    h, w = mask.shape
    ys = np.arange(h)
    band = np.abs(y_to_lat(ys, h)) <= math.radians(90 - cap_deg)
    holes = (mask == 0) & band[:, None]
    n = int(holes.sum())
    if n == 0:
        return pano, 0
    rows = np.where(holes.any(axis=1))[0]
    y0, y1 = max(0, int(rows.min()) - margin), min(h, int(rows.max()) + 1 + margin)
    sub = cv2.copyMakeBorder(pano[y0:y1], 0, 0, margin, margin, cv2.BORDER_WRAP)
    hs = cv2.copyMakeBorder(holes[y0:y1].astype(np.uint8), 0, 0, margin, margin, cv2.BORDER_WRAP) > 0
    filled = np.clip(_pull_push(sub, ~hs), 0, 255)[:, margin:margin + w]
    soft = cv2.GaussianBlur(holes[y0:y1].astype(np.float32), (0, 0), 3)
    soft = np.maximum(soft, holes[y0:y1].astype(np.float32))[..., None]
    out = pano.copy()
    out[y0:y1] = (filled * soft + pano[y0:y1] * (1 - soft)).astype(np.uint8)
    return out, n


# ---------------------------------------------------------------------------------------------- metadata ---
def gpano_xmp(w: int, h: int, photos: int) -> bytes:
    return (
        '<x:xmpmeta xmlns:x="adobe:ns:meta/"><rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description rdf:about="" xmlns:GPano="http://ns.google.com/photos/1.0/panorama/" '
        'GPano:ProjectionType="equirectangular" GPano:UsePanoramaViewer="True" '
        f'GPano:CroppedAreaImageWidthPixels="{w}" GPano:CroppedAreaImageHeightPixels="{h}" '
        f'GPano:FullPanoWidthPixels="{w}" GPano:FullPanoHeightPixels="{h}" '
        'GPano:CroppedAreaLeftPixels="0" GPano:CroppedAreaTopPixels="0" GPano:PoseHeadingDegrees="0" '
        f'GPano:SourcePhotosCount="{photos}" GPano:CaptureSoftware="Reachy Gemini DJ"/>'
        '</rdf:RDF></x:xmpmeta>').encode()


def jpeg_with_xmp(jpeg: bytes, xmp: bytes) -> bytes:
    """Insert an XMP APP1 segment after SOI (and after a JFIF APP0 if there is one)."""
    if jpeg[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG")
    body = b"http://ns.adobe.com/xap/1.0/\x00" + xmp
    seg = b"\xff\xe1" + (len(body) + 2).to_bytes(2, "big") + body
    pos = 2
    if jpeg[2:4] == b"\xff\xe0":
        pos = 4 + int.from_bytes(jpeg[4:6], "big")
    return jpeg[:pos] + seg + jpeg[pos:]


# --------------------------------------------------------------------------------------------- the stitch ---
def available_mb() -> float | None:
    """Memory the system could give a new allocation (MB), or None when /proc/meminfo is unreadable."""
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def pick_out_w(frame_w: int, avail_mb: float | None = None, margin_mb: float = 1000.0, n_frames: int = 40) -> int:
    """Equirect width for the stitch: matched to the photos' detail, and no bigger than the memory allows (an
    out-of-memory kill would take the robot's app with it). Needs the estimated peak plus margin_mb free."""
    if frame_w < 2000:
        return OUT_W_STREAM
    avail = available_mb() if avail_mb is None else avail_mb
    if avail is None:
        return OUT_W_STEPS[1]
    for w in OUT_W_STEPS:
        if 450 + 2000 * (w / 8192) ** 2 + 6 * max(0, n_frames - 40) + margin_mb <= avail:
            return w
    return OUT_W_STEPS[-1]


def stitch(frames: list[Frame], cam: Camera | None = None, out_w: int | None = None, progress=lambda f, s: None,
           do_refine: bool = True, matches=None) -> tuple[np.ndarray, dict]:
    """Photos -> equirectangular panorama. matches: from a Matcher fed during the capture (else matched here)."""
    import cv2
    t0 = time.monotonic()
    cam = cam or Camera(size=frames[0].size)
    out_w = out_w or pick_out_w(frames[0].size[0], n_frames=len(frames))
    w, h = out_w, out_w // 2
    pad = w // 8
    info: dict = {"frames": len(frames), "out_w": out_w}
    scale = (1.0, 1.0)
    if do_refine and len(frames) > 1:
        progress(0.05, "Matching photos")
        if matches is None:
            matches = match_frames(cam, frames)
        progress(0.25, "Aligning")
        info["align"] = refine(cam, frames, matches)
    info["t_align"] = round(time.monotonic() - t0, 2)
    progress(0.35, "Matching colours")
    gains = solve_gains(cam, frames)
    info["gains"] = np.round(gains, 3).tolist()
    progress(0.45, "Projecting")
    pieces: list[Warp] = []
    for k, f in enumerate(frames):
        pieces.extend(warp_frame(cam, f.load(), f.R, w, h, pad, scale, gains[k]))
    info["t_warp"] = round(time.monotonic() - t0, 2)
    progress(0.6, "Blending")
    canvas, cmask = blend(pieces, w, h, pad)
    del pieces
    pano, pmask = fold_wrap(canvas, cmask, w, pad)
    del canvas, cmask
    info["t_blend"] = round(time.monotonic() - t0, 2)
    progress(0.9, "Filling the top and bottom")
    pano, poles = fill_poles(pano, pmask)
    info["poles"] = poles
    pano, info["gap_px"] = fill_gaps(pano, pmask)
    info["coverage"] = round(float((pmask > 0).mean()), 4)
    info["t_total"] = round(time.monotonic() - t0, 2)
    return pano, info


def coverage(shots: list[Shot], cam: Camera, w: int = 360) -> dict:
    """Fraction of the sphere a plan sees with nominal poses (for planning; no images needed)."""
    h = w // 2
    lat = y_to_lat(np.arange(h), h)[:, None]
    lon = x_to_lon(np.arange(w), w)[None, :]
    dx, dy, dz = lonlat_to_dir(lon, lat)
    dz = np.broadcast_to(dz, dx.shape)
    seen = np.zeros((h, w), bool)
    for s in shots:
        R = rot_xyz(s.roll, s.pitch, s.yaw) @ expmap(MOUNT_ROTVEC) @ CAM_IN_HEAD
        cx = R[0, 0] * dx + R[1, 0] * dy + R[2, 0] * dz
        cy = R[0, 1] * dx + R[1, 1] * dy + R[2, 1] * dz
        cz = R[0, 2] * dx + R[1, 2] * dy + R[2, 2] * dz
        ok = cz > 1e-3
        czs = np.where(ok, cz, 1)
        seen |= ok & cam.inside(cx / czs, cy / czs)
    area = np.cos(lat) * np.ones((1, w))
    rows = seen.any(axis=1)
    top = float(np.degrees(lat[rows.argmax(), 0])) if rows.any() else None
    full_rows = seen.all(axis=1)
    return {"sphere": float((seen * area).sum() / area.sum()),
            "full_ring_up_to": float(np.degrees(lat[full_rows, 0].max())) if full_rows.any() else None,
            "full_ring_down_to": float(np.degrees(lat[full_rows, 0].min())) if full_rows.any() else None,
            "highest_seen": top}


def encode(pano: np.ndarray, photos: int, quality: int = JPEG_QUALITY) -> tuple[bytes, bytes]:
    """(full JPEG with GPano XMP, 640x320 thumbnail JPEG)."""
    import cv2
    h, w = pano.shape[:2]
    ok, buf = cv2.imencode(".jpg", pano, [cv2.IMWRITE_JPEG_QUALITY, quality, cv2.IMWRITE_JPEG_OPTIMIZE, 1])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    thumb = cv2.resize(pano, (640, 320), interpolation=cv2.INTER_AREA)
    ok2, tb = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 82])
    return jpeg_with_xmp(buf.tobytes(), gpano_xmp(w, h, photos)), tb.tobytes()


# ------------------------------------------------------------------------------------- full-resolution camera ---
STILL_SIZE = (3072, 1728)       # libcamera picks the full-resolution 4608x2592 sensor mode: the lens's whole field of view
STILL_FPS = 10
# Continuous autofocus: in a room the robot sees things from 30 cm to several metres, and no single lens position is
# sharp everywhere (measured: up to 4x sharper than any fixed focus in close-up directions). grab() waits for the
# picture to settle (no motion blur, focus done) before it takes one.
STILL_CONTROLS = "af-mode=continuous"
STILL_MIN_FRESH = 3             # frames always skipped after the head stops (1-2 still show the end of the move)
STILL_SETTLE_S = 1.5            # at most this long waiting for the sharpness to settle
STILL_PLATEAU = 0.04            # settled = three frames in a row within 4 % sharpness of each other
DAEMON_URL = "http://127.0.0.1:8000"


class StillCamera:
    """Full-resolution, full-field-of-view frames straight from libcamera.

    The daemon's own stream is 1280x720 cut from the middle 2/3 of the sensor (Camera Module 3 Wide, IMX708). For the
    photo sphere the app asks the daemon to release the camera (POST /api/media/release: live video pauses), opens
    libcamera itself at STILL_SIZE, and always gives the camera back (POST /api/media/acquire) when done. Use as a
    context manager; grab() returns a BGR frame exposed after the call (older buffered frames are skipped)."""

    def __init__(self, mini, size=STILL_SIZE, fps: int = STILL_FPS, daemon: str = DAEMON_URL,
                 controls: str = STILL_CONTROLS):
        self.mini, self.size, self.fps, self.daemon, self.controls = mini, tuple(size), fps, daemon, controls
        self.last = {}                               # how the last grab() went: frames, wait, sharpness
        self.pipe = self.sink = None
        self._released = self._reader_closed = False

    def _post(self, action: str) -> None:
        import urllib.request
        req = urllib.request.Request(f"{self.daemon}/api/media/{action}", data=b"", method="POST")
        with urllib.request.urlopen(req, timeout=20) as r:
            r.read()

    def __enter__(self) -> "StillCamera":
        try:
            import gi
            gi.require_version("Gst", "1.0")
            from gi.repository import Gst
            Gst.init(None)
            cam = getattr(getattr(self.mini, "media", None), "camera", None)
            if cam is not None and hasattr(cam, "close"):
                cam.close()                          # the app's IPC reader, before the daemon's socket goes away
                self._reader_closed = True
            self._post("release")
            self._released = True
            w, h = self.size
            self.pipe = Gst.parse_launch(
                f"libcamerasrc {self.controls} ! video/x-raw,width={w},height={h},framerate={self.fps}/1,format=NV12 ! "
                "appsink name=sink drop=true max-buffers=1 sync=false")
            self.pipe.set_state(Gst.State.PLAYING)
            self.sink = self.pipe.get_by_name("sink")
            end = time.monotonic() + 6.0             # first frame, then let auto exposure / white balance settle
            got = 0
            while got < 12 and time.monotonic() < end:
                got += self._pull(1.0) is not None
            if got == 0:
                raise RuntimeError("no frames from the full-resolution camera")
            return self
        except Exception:
            self.__exit__(None, None, None)
            raise

    def _pull(self, timeout_s: float):
        """The next frame as NV12 (h*3/2, w) uint8, or None."""
        from gi.repository import Gst
        sample = self.sink.emit("try-pull-sample", int(timeout_s * 1e9))
        if sample is None:
            return None
        w, h = self.size
        buf = sample.get_buffer()
        ok, info = buf.map(Gst.MapFlags.READ)
        if not ok:
            return None
        try:
            stride = len(info.data) * 2 // (3 * h)   # NV12 rows may be padded
            return np.array(np.frombuffer(info.data, np.uint8)[: stride * h * 3 // 2].reshape(h * 3 // 2, stride)[:, :w])
        finally:
            buf.unmap(info)

    @staticmethod
    def sharpness(nv12, h: int) -> float:
        """Laplacian variance of the luma at 1/4 scale: drops with motion blur and with defocus."""
        import cv2
        return float(cv2.Laplacian(nv12[:h:4, ::4], cv2.CV_32F).var())

    def grab(self, min_fresh: int = STILL_MIN_FRESH, settle_s: float = STILL_SETTLE_S, timeout_s: float = 3.0):
        """A BGR frame taken after the head stopped: skips min_fresh frames, then waits (up to settle_s) until the
        sharpness stops changing (motion blur gone, autofocus done) and returns the sharpest of the settled frames."""
        import cv2
        h = self.size[1]
        t0 = time.monotonic()
        for _ in range(max(1, min_fresh)):
            f = self._pull(timeout_s)
            if f is None:
                return None
        hist = [(self.sharpness(f, h), f)]
        while time.monotonic() - t0 < settle_s:
            vals = [v for v, _ in hist[-3:]]
            if len(vals) == 3 and max(vals) - min(vals) <= STILL_PLATEAU * max(vals):
                break
            f = self._pull(timeout_s)
            if f is None:
                break
            hist.append((self.sharpness(f, h), f))
        best_v, best = max(hist[-3:], key=lambda t: t[0])
        self.last = {"frames": min_fresh + len(hist) - 1, "wait_s": round(time.monotonic() - t0, 2),
                     "sharpness": round(best_v, 1)}
        return cv2.cvtColor(best, cv2.COLOR_YUV2BGR_NV12)

    def __exit__(self, *exc) -> None:
        if self.pipe is not None:
            try:
                from gi.repository import Gst
                self.pipe.set_state(Gst.State.NULL)
            except Exception as e:
                log.warning("stopping the full-resolution camera: %s", e)
            self.pipe = self.sink = None
        if self._released:
            for attempt in range(3):
                try:
                    self._post("acquire")
                    break
                except Exception as e:
                    log.warning("giving the camera back to the daemon failed (%s), retrying", e)
                    time.sleep(1.0)
            self._released = False
            time.sleep(1.0)                          # the daemon's IPC socket comes back with its pipeline
        if self._reader_closed:
            try:
                self.mini.media.camera.open()
            except Exception as e:
                log.warning("reopening the app's camera reader: %s", e)
            self._reader_closed = False


# ------------------------------------------------------------------------------------------- the robot job ---
ID_RE = __import__("re").compile(r"^\d{8}-\d{6}$")
SWING_DEG_PER_S = 45.0         # the big turns to and from the first photo (body up to 150 degrees): slower still
MOVE_DEG_PER_S = 60.0          # goto duration from the largest joint change (smooth min-jerk move; slow and gentle on the motors)
STILL_DEG = 0.06               # head counts as still when two pose reads 40 ms apart differ by less than this
POINT_TOL_DEG = 2.5             # re-aim a shot when the head lands further than this from where it should look
POINT_FIXES = 2                 # at most this many corrective nudges per shot
HEAD_TURN_LIMIT = 20.0          # head yaw relative to the body, degrees (kept small: the neck cables)
FRESH_FRAMES = 3               # frames to skip after the head stops (camera pipeline latency ~50-100 ms)


class PanoramaJob:
    """One capture + stitch at a time, run on a worker thread. The page polls status(); files land in store/."""

    def __init__(self, get_mini, motion=None, store: Path | None = None, on_event=lambda kind, msg: None,
                 motion_enabled=lambda: True):
        from .config import HOME
        self.get_mini, self.motion, self.on_event, self.motion_enabled = get_mini, motion, on_event, motion_enabled
        self.store = Path(store or HOME / "panoramas")
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._st = {"state": "idle", "progress": 0.0, "step": "", "id": None, "error": None, "eta_s": None}
        try:
            self._tidy_raw()
        except Exception as e:
            log.warning("could not tidy the raw photos: %s", e)
        self.full_res = True             # full sensor via the daemon's media release (live video pauses meanwhile)
        self.source = "stream"
        self.mode = DEFAULT_MODE
        self.grabs: list[dict] = []

    # ------------------------------------------------------------------ API used by main.py / the voice tool
    def status(self) -> dict:
        with self._lock:
            return dict(self._st)

    def busy(self) -> bool:
        return self.status()["state"] in ("capturing", "stitching")

    def start(self, source: str = "page", mode: str | None = None) -> dict:
        given = mode
        mode = normalise_mode(mode)
        if mode is None:
            return {"ok": False, "error": f"Unknown mode {given!r}. Use one of: {', '.join(MODES)}."}
        with self._lock:
            if self._st["state"] in ("capturing", "stitching"):
                return {"ok": False, "error": "A 360° photo is already being taken."}
            mini = self.get_mini()
            if mini is None:
                return {"ok": False, "error": "The robot is not connected yet."}
            if not self.motion_enabled():
                return {"ok": False, "error": "Movement is turned off in the settings."}
            if self.motion is not None and getattr(self.motion, "learning", False):
                return {"ok": False, "error": "Learn mode is on (motors relaxed). Turn it off first."}
            pid = datetime.now().strftime("%Y%m%d-%H%M%S")
            self._cancel.clear()
            self._st = {"state": "capturing", "progress": 0.0, "step": "Getting ready", "id": pid, "error": None,
                        "eta_s": None}
        self._thread = threading.Thread(target=self._run, args=(pid, source, mode), daemon=True, name="panorama")
        self._thread.start()
        return {"ok": True, "id": pid}

    def cancel(self) -> dict:
        self._cancel.set()
        return {"ok": True}

    def list(self) -> list[dict]:
        items = []
        if self.store.exists():
            for meta in sorted(self.store.glob("*.json"), reverse=True):
                try:
                    m = json.loads(meta.read_text())
                    pid = meta.stem
                    if not ID_RE.match(pid) or not (self.store / f"{pid}.jpg").exists():
                        continue
                    items.append({"id": pid, "created": m.get("created"), "width": m.get("width"), "height": m.get("height"),
                                  "frames": m.get("frames"), "seconds": m.get("seconds"),
                                  "size_bytes": (self.store / f"{pid}.jpg").stat().st_size,
                                  "url": f"/api/panorama/{pid}.jpg", "thumb_url": f"/api/panorama/{pid}/thumb.jpg"})
                except Exception:
                    continue
        return items

    def path(self, pid: str, thumb: bool = False) -> Path | None:
        if not ID_RE.match(pid or ""):
            return None
        p = self.store / (f"{pid}_thumb.jpg" if thumb else f"{pid}.jpg")
        return p if p.exists() else None

    def delete(self, pid: str) -> bool:
        """Remove a 360 photo completely: the picture, its thumbnail and details, and (when it is the capture the
        raw photos in last_raw came from, or no photo is left) those raw photos too."""
        if not ID_RE.match(pid or ""):
            return False
        found = False
        for suffix in (".jpg", "_thumb.jpg", ".json"):
            p = self.store / f"{pid}{suffix}"
            if p.exists():
                p.unlink()
                found = True
        raw = self.store / "last_raw"
        if raw.exists():
            try:
                owner = (raw / "id.txt").read_text().strip()
            except OSError:
                owner = ""
            if owner == pid or not any(self.store.glob("*.json")):
                shutil.rmtree(raw, ignore_errors=True)
        return found

    # ------------------------------------------------------------------ worker
    def _set(self, **kw) -> None:
        with self._lock:
            self._st.update(kw)

    def _run(self, pid: str, source: str, mode: str = DEFAULT_MODE) -> None:
        self.mode = mode
        t0 = time.monotonic()
        held = False
        frames: list[Frame] = []
        try:
            mini = self.get_mini()
            if self.motion is not None and hasattr(self.motion, "hold_begin"):
                self.motion.hold_begin("360 photo")
                held = True
            self.on_event("camera", f"Taking a 360° photo ({source})")
            frames = self._capture(mini, t0, mode)
            if held:
                self.motion.hold_end()
                held = False
            if self._cancel.is_set():
                self._set(state="idle", step="Cancelled", progress=0.0, eta_s=None)
                self.on_event("camera", "360° photo cancelled")
                return
            self._set(state="stitching", step="Stitching", progress=0.0, eta_s=None)
            t1 = time.monotonic()
            self._set(step="Matching photos")
            matches = self._capture_matches(len(frames))
            self._save_raw(frames, pid)
            pano, info = stitch(frames, progress=lambda f, s: self._set(progress=round(f, 3), step=s), matches=matches)
            info["matched_during_capture"] = matches is not None
            self._set(step="Saving", progress=0.97)
            full, thumb = encode(pano, len(frames))
            self.store.mkdir(parents=True, exist_ok=True)
            (self.store / f"{pid}.jpg").write_bytes(full)
            (self.store / f"{pid}_thumb.jpg").write_bytes(thumb)
            meta = {"created": datetime.now().isoformat(timespec="seconds"), "width": pano.shape[1], "height": pano.shape[0],
                    "frames": len(frames), "mode": self.mode, "source": self.source, "seconds": round(time.monotonic() - t0, 1),
                    "capture_s": round(t1 - t0, 1), "stitch_s": round(time.monotonic() - t1, 1),
                    "info": {k: v for k, v in info.items() if k != "gains"}}
            if self.grabs:
                meta["grabs"] = {"wait_s_mean": round(float(np.mean([g["wait_s"] for g in self.grabs])), 2),
                                 "sharpness_min": min(g["sharpness"] for g in self.grabs),
                                 "sharpness_median": float(np.median([g["sharpness"] for g in self.grabs]))}
            (self.store / f"{pid}.json").write_text(json.dumps(meta, indent=1, default=float))
            self._prune()
            log.info("360 photo %s: %d photos, capture %.1f s, stitch %.1f s, align %s", pid, len(frames),
                     meta["capture_s"], meta["stitch_s"], info.get("align"))
            self.on_event("camera", f"360° photo saved ({len(frames)} photos, {meta['seconds']:.0f} s)")
            self._set(state="done", step="Done", progress=1.0, eta_s=0)
        except Exception as e:
            log.exception("360 photo failed")
            self._drop_raw(pid)                          # no photo came out of it: its raw frames are of no use
            self._set(state="error", error=f"{type(e).__name__}: {e}"[:200], step="Failed", eta_s=None)
            self.on_event("error", f"360° photo failed: {e}")
        finally:
            if held:
                try:
                    self.motion.hold_end()
                except Exception:
                    pass

    def _raw_owner(self) -> str | None:
        """ID of the capture the raw frames in last_raw came from (None: no raw frames, or from before ids were kept)."""
        try:
            return (self.store / "last_raw" / "id.txt").read_text().strip() or None
        except OSError:
            return None

    def _drop_raw(self, pid: str) -> None:
        """Remove the raw frames if they belong to this capture and it has no finished photo."""
        if self._raw_owner() == pid and not (self.store / f"{pid}.json").exists():
            shutil.rmtree(self.store / "last_raw", ignore_errors=True)

    def _tidy_raw(self) -> None:
        """At start-up: remove raw frames left by a capture that never finished (the app stopped mid-stitch), or that
        remain when no photo is left."""
        raw = self.store / "last_raw"
        if not raw.exists():
            return
        owner = self._raw_owner()
        orphan = owner is not None and not (self.store / f"{owner}.json").exists()
        if orphan or not any(self.store.glob("*.json")):
            shutil.rmtree(raw, ignore_errors=True)

    def _save_raw(self, frames: list[Frame], pid: str = "") -> None:
        """Keep the last capture's photos and poses (store/last_raw, replaced each time) so it can be re-stitched.
        Deleting that 360 photo removes them too (see delete)."""
        import cv2
        raw = self.store / "last_raw"
        try:
            shutil.rmtree(raw, ignore_errors=True)
            raw.mkdir(parents=True)
            for k, f in enumerate(frames):
                if f.jpeg is not None:
                    (raw / f"{k:02d}.jpg").write_bytes(f.jpeg)
                else:
                    cv2.imwrite(str(raw / f"{k:02d}.jpg"), f.image, [cv2.IMWRITE_JPEG_QUALITY, 92])
            (raw / "frames.json").write_text(json.dumps([{"i": k, "pose": np.asarray(f.pose).tolist()}
                                                         for k, f in enumerate(frames)]))
            (raw / "id.txt").write_text(pid)
        except Exception as e:                       # never let this break the photo
            log.warning("could not keep the raw photos: %s", e)

    def _prune(self) -> None:
        metas = sorted(self.store.glob("*.json"))
        for m in metas[:-KEEP] if len(metas) > KEEP else []:
            self.delete(m.stem)

    # ------------------------------------------------------------------ capture
    def _capture(self, mini, t0: float, mode: str = DEFAULT_MODE) -> list[Frame]:
        from reachy_mini.utils import create_head_pose
        shots = plan(mode)
        frames: list[Frame] = []
        try:
            mini.enable_motors()
        except Exception as e:
            log.warning("enable_motors: %s", e)
        start_pose = np.asarray(mini.get_current_head_pose())
        work = self._start_matcher()
        still = None
        if self.full_res:
            self._set(step="Getting the camera")
            try:
                still = StillCamera(mini).__enter__()
            except Exception as e:
                log.warning("full-resolution camera unavailable (%s); using the video stream", e)
                still = None
        self.source = f"{STILL_SIZE[0]}x{STILL_SIZE[1]}" if still else "stream"
        grabs: list[dict] = []
        self.grabs = grabs
        try:
            prev = Shot(0.0, 0.0)
            per_shot = None
            self._drain(mini)
            first = shots[0]                  # swing round to the first photo with the head level, tilt and roll afterwards
            self._goto(mini, create_head_pose(roll=0, pitch=0, yaw=first.yaw, degrees=True), math.radians(first.body),
                       max(1.0, abs(first.body) / SWING_DEG_PER_S))
            for k, s in enumerate(shots):
                if self._cancel.is_set():
                    break
                ts = time.monotonic()
                change = max(abs(s.yaw - prev.yaw), abs(s.body - prev.body), abs(s.pitch - prev.pitch), abs(s.roll - prev.roll))
                dur = max(0.6, change / MOVE_DEG_PER_S)
                self._point(mini, s, dur, create_head_pose)
                img = still.grab() if still else self._fresh_frame(mini)
                if img is None:
                    raise RuntimeError("The camera stopped sending pictures.")
                frames.append(Frame(img, np.asarray(mini.get_current_head_pose(), dtype=np.float64)))
                if still is not None:
                    grabs.append(still.last)
                work.put(frames[-1])
                prev = s
                dt = time.monotonic() - ts
                per_shot = dt if per_shot is None else 0.7 * per_shot + 0.3 * dt
                left = len(shots) - k - 1
                self._set(progress=round((k + 1) / len(shots), 3), step=f"Photo {k + 1} of {len(shots)}",
                          eta_s=round(left * per_shot + 3.6 * len(shots) + 10, 0))
        finally:
            work.put(None)
            if still is not None:
                still.__exit__(None, None, None)     # live video comes back
        # back to where it started (body centred): level the head first, then swing the body round
        try:
            self._goto(mini, create_head_pose(roll=0, pitch=0, yaw=prev.yaw, degrees=True), math.radians(prev.body), 1.2)
            self._goto(mini, start_pose, 0.0, max(1.5, abs(prev.body) / SWING_DEG_PER_S))
        except Exception as e:
            log.warning("could not return after the 360 photo: %s", e)
        return frames

    def _start_matcher(self) -> "queue.Queue":
        """Background thread that matches each photo against the earlier ones while the robot moves on."""
        work: queue.Queue = queue.Queue()
        self._matcher, self._matcher_ok = None, threading.Event()

        def run():
            try:
                while (f := work.get()) is not None:
                    if self._matcher is None:
                        self._matcher = Matcher(Camera(size=f.size))
                    self._matcher.add(f)
                    if f.size[0] > 2000:                 # full-resolution photo: keep it compressed (40 x 16 MB)
                        f.compress()
                self._matcher_ok.set()
            except Exception:
                log.exception("matching during the capture failed; it will be redone while stitching")
                self._matcher = None
                while work.get() is not None:   # drain so the capture never blocks
                    pass

        self._matcher_thread = threading.Thread(target=run, name="pano-match", daemon=True)
        self._matcher_thread.start()
        return work

    def _capture_matches(self, n_frames: int):
        """Matches made during the capture (waits for the last photo), or None to match while stitching."""
        t = getattr(self, "_matcher_thread", None)
        if t is None:
            return None
        t.join(timeout=120)
        m = self._matcher
        if not self._matcher_ok.is_set() or m is None or len(m.feats) != n_frames:
            return None
        return m.matches

    def _point(self, mini, s: Shot, duration: float, create_head_pose) -> None:
        """Move to a shot and make sure the head really got there. The head stops up to ~12 degrees short in pitch
        (it lags in the direction it came from) and the body ~8-18 degrees short near +-160, but its own pose reading
        agrees with the photos to ~3 degrees, so read it back and nudge the command until it is close."""
        yaw, pitch, body = s.yaw, s.pitch, s.body
        for attempt in range(1 + POINT_FIXES):
            self._goto(mini, create_head_pose(roll=s.roll, pitch=pitch, yaw=yaw, degrees=True), math.radians(body),
                       duration if attempt == 0 else 0.3)
            self._wait_still(mini)
            if attempt == POINT_FIXES:
                return
            my, mp, _ = head_ypr(mini.get_current_head_pose())
            ey = (s.yaw - my + 180) % 360 - 180
            ep = 0.0 if s.roll else s.pitch - mp         # rolled shots already sit at the edge of what IK allows
            if abs(ey) < POINT_TOL_DEG and abs(ep) < POINT_TOL_DEG:
                return
            nb = max(-BODY_LIMIT, min(BODY_LIMIT, body + ey))
            ny = max(nb - HEAD_TURN_LIMIT, min(nb + HEAD_TURN_LIMIT, yaw + ey))
            npitch = max(-PITCH_LIMIT, min(PITCH_LIMIT, pitch + ep))
            if max(abs(nb - body), abs(ny - yaw), abs(npitch - pitch)) < 0.5:
                return                                   # already asking for the limit: this is as far as it goes
            yaw, pitch, body = ny, npitch, nb

    @staticmethod
    def _goto(mini, head, body_yaw: float, duration: float) -> None:
        for attempt in range(20):
            try:
                mini.goto_target(head=head, body_yaw=body_yaw, duration=duration)
                return
            except Exception as e:
                msg = str(e).lower()
                if "busy" in msg or "already running" in msg:        # MoveBusyError: another move is finishing
                    time.sleep(0.1)
                elif "out of range" in msg and attempt < 3:          # the daemon stalled mid-move (e.g. its relay
                    log.warning("move failed (%s); trying it again", e)   # reconnecting): a retry is a fresh move
                    time.sleep(0.4)
                elif "communication error" in msg and attempt < 4:   # a brief serial glitch to one motor: the link
                    log.warning("move failed (%s); trying it again", e)   # recovers by itself within a second or two
                    time.sleep(1.5)
                else:
                    raise
        raise RuntimeError("The robot was busy with another move.")

    @staticmethod
    def _wait_still(mini, timeout: float = 1.5) -> None:
        end = time.monotonic() + timeout
        last = np.asarray(mini.get_current_head_pose())[:3, :3]
        calm = 0
        while time.monotonic() < end:
            time.sleep(0.04)
            cur = np.asarray(mini.get_current_head_pose())[:3, :3]
            ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(cur @ last.T) - 1) / 2))))
            last = cur
            calm = calm + 1 if ang < STILL_DEG else 0
            if calm >= 2:
                return

    @staticmethod
    def _drain(mini) -> None:
        for _ in range(10):
            if mini.media.get_frame() is None:
                return

    @staticmethod
    def _fresh_frame(mini, n: int = FRESH_FRAMES, timeout: float = 2.0):
        """get_frame() returns None until a new frame arrives; skip n-1 new frames, return the n-th."""
        got, f, end = 0, None, time.monotonic() + timeout
        while got < n and time.monotonic() < end:
            g = mini.media.get_frame()
            if g is not None:
                got, f = got + 1, g
            else:
                time.sleep(0.004)
        return f
