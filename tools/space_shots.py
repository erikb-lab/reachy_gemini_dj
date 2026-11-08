#!/usr/bin/env python3
"""Print, as JSON, where each photo of each 360 mode lands on the sphere: one outline per photo, as [lon, lat] degrees.
Used to draw the mode picker on the Space page (index.html). Run from the repo root with the project's venv:
    .venv/bin/python tools/space_shots.py > /tmp/x.json   (then paste into index.html's SHOTS constant)"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np

from reachy_gemini_dj import panorama as P

cam = P.Camera(size=(3072, 1728))
x0, x1, y0, y1 = cam.box
N = 96
gx, gy = np.meshgrid(np.linspace(x0, x1, N), np.linspace(y0, y1, int(N * (y1 - y0) / (x1 - x0))))
inside = cam.inside(gx, gy)
cs, _ = cv2.findContours(inside.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
c = max(cs, key=cv2.contourArea)
c = c.reshape(-1, 2)
c = c[np.linspace(0, len(c), 24, endpoint=False).astype(int)]     # ~24 points: great-circle edges bend on the map
nx = x0 + c[:, 0] / (N - 1) * (x1 - x0)
ny = y0 + c[:, 1] / (gx.shape[0] - 1) * (y1 - y0)
rays0 = np.stack([nx, ny, np.ones_like(nx)], 1)
rays0 /= np.linalg.norm(rays0, axis=1, keepdims=True)

out = {}
for mode in P.MODES:
    polys = []
    for s in P.plan(mode):
        R = P.rot_xyz(s.roll, s.pitch, s.yaw) @ P.expmap(P.MOUNT_ROTVEC) @ P.CAM_IN_HEAD
        v = rays0 @ R.T
        lon = np.degrees(np.arctan2(v[:, 1], v[:, 0]))
        lat = np.degrees(np.arcsin(np.clip(v[:, 2], -1, 1)))
        centre = np.degrees(math.atan2(*(R[:, 2][[1, 0]])))
        lon = centre + (lon - centre + 180) % 360 - 180          # unwrap around the photo's centre
        polys.append([[round(float(a)), round(float(b))] for a, b in zip(lon, lat)])
    out[mode] = polys
print(json.dumps(out, separators=(",", ":")))
