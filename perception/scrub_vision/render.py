"""Tiny top-down rasteriser for synthetic piles: perspective from a downward camera, per-instance
painter's order (bottom to top), back-face culling, Lambert shading, soft drop shadows.

Scene units are mm. Camera looks straight down the -z axis from `height_mm` above the table plane z=0.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import trimesh

from .templates import Instrument, fill_tris, mesh_for



def random_filament_bgr(rng: np.random.Generator) -> np.ndarray:
    """Any print colour: saturated hues of every kind plus whites, greys, silvers and blacks."""
    if rng.random() < 0.2:
        v = rng.uniform(25, 245)
        return np.array([v, v, v], np.float32) + rng.normal(0, 4, 3)
    hsv = np.uint8([[[rng.integers(0, 180), rng.integers(70, 256), rng.integers(60, 241)]]])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0].astype(np.float32)


@dataclass
class CamSpec:
    w: int = 960
    h: int = 540
    fx: float = 700.0          # px; ~550 mm wide view at 400 mm height
    height_mm: float = 400.0
    cx: float | None = None
    cy: float | None = None

    def project(self, xyz: np.ndarray) -> np.ndarray:
        cx = self.w / 2 if self.cx is None else self.cx
        cy = self.h / 2 if self.cy is None else self.cy
        depth = np.maximum(self.height_mm - xyz[:, 2], 1.0)
        return np.c_[cx + self.fx * xyz[:, 0] / depth, cy + self.fx * xyz[:, 1] / depth]


_scene_cache: dict[str, trimesh.Trimesh] = {}


def scene_mesh(inst: Instrument) -> trimesh.Trimesh:
    """Flat mesh (tip at +x) centred on its bounding box, so z spans [-t/2, t/2]."""
    if inst.id not in _scene_cache:
        m = mesh_for(inst).copy()
        m.vertices -= m.bounds.mean(0)
        _scene_cache[inst.id] = m
    return _scene_cache[inst.id]


def pose(x: float, y: float, z: float, yaw: float, flip: bool = False) -> np.ndarray:
    """4x4: optional flip (pi about x: lies on its other face), then yaw about z, then translate."""
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    if flip:
        R = R @ np.diag([1.0, -1.0, -1.0])
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, (x, y, z)
    return T


def _noise_bg(cam: CamSpec, rng: np.random.Generator) -> np.ndarray:
    base = rng.uniform(40, 245)            # dark mats to white sheets
    tint = rng.uniform(-25, 25, 3)
    low = cv2.resize(rng.normal(0, 1, (6, 10)).astype(np.float32), (cam.w, cam.h), interpolation=cv2.INTER_CUBIC)
    img = np.clip(base + tint + low[..., None] * rng.uniform(4, 18), 0, 255)
    for _ in range(rng.integers(0, 6)):        # sheet wrinkles
        p0 = rng.uniform([0, 0], [cam.w, cam.h]).astype(int)
        p1 = (p0 + rng.normal(0, 200, 2)).astype(int)
        layer = np.zeros((cam.h, cam.w), np.float32)
        cv2.line(layer, tuple(map(int, p0)), tuple(map(int, p1)), 1.0, int(rng.integers(2, 8)))
        img += (cv2.GaussianBlur(layer, (0, 0), rng.uniform(2, 6)) * rng.uniform(-25, 15))[..., None]
    yy, xx = np.mgrid[0:cam.h, 0:cam.w]
    vign = 1 - rng.uniform(0, 0.25) * (((xx - cam.w / 2) / cam.w) ** 2 + ((yy - cam.h / 2) / cam.h) ** 2) * 2
    return np.clip(img * vign[..., None], 0, 255).astype(np.float32)


def render_scene(items: list[tuple[trimesh.Trimesh, np.ndarray]], cam: CamSpec, rng: np.random.Generator,
                 bg: np.ndarray | None = None, colours: list | None = None) -> tuple[np.ndarray, list[np.ndarray]]:
    """Render meshes at 4x4 poses. Returns BGR uint8 image and one bool mask per item (after occlusion)."""
    img = (_noise_bg(cam, rng) if bg is None else bg.astype(np.float32)).copy()
    ids = np.zeros((cam.h, cam.w), np.int32)
    light = rng.normal([0.3, -0.2, 1.0], 0.25)
    light /= np.linalg.norm(light)
    shadow_off = -light[:2] / max(light[2], 0.3) * rng.uniform(2, 6)
    # A set is printed in one filament (any colour, different between sets); individual prints vary
    # slightly in shade, so nothing in a pile can be told apart by colour.
    shared = random_filament_bgr(rng)
    order = sorted(range(len(items)), key=lambda k: items[k][1][2, 3])
    for k in order:
        mesh, T = items[k]
        base = (np.array(colours[k], np.float32) if colours else
                np.clip(shared * rng.uniform(0.93, 1.07) + rng.normal(0, 3, 3), 0, 255))
        v = mesh.vertices @ T[:3, :3].T + T[:3, 3]
        tri = v[mesh.faces]
        n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        nn = np.linalg.norm(n, axis=1)
        keep = (n[:, 2] > 0) & (nn > 1e-9)       # facing the camera
        uv = cam.project(v)
        polys = uv[mesh.faces[keep]]
        shade = 0.45 + 0.55 * np.clip((n[keep] / nn[keep, None]) @ light, 0, 1)
        # soft drop shadow on what's below
        sil = np.zeros((cam.h, cam.w), np.uint8)
        fill_tris(sil, polys + shadow_off * cam.fx / cam.height_mm, 255)
        sh = cv2.GaussianBlur(sil.astype(np.float32) / 255, (0, 0), 3) * rng.uniform(0.15, 0.35)
        img *= (1 - sh)[..., None]
        # draw faces lowest-first so an instrument's raised parts cover its own lower parts
        zf = v[mesh.faces[keep]][:, :, 2].mean(1)
        for i in np.argsort(zf):
            col = tuple(float(c) for c in np.clip(base * shade[i] * 1.1, 0, 255))
            cv2.fillConvexPoly(img, np.round(polys[i]).astype(np.int32), col)
        fill_tris(ids, polys, k + 1)
    masks = [ids == k + 1 for k in range(len(items))]
    return np.clip(img, 0, 255).astype(np.uint8), masks


def add_gripper(img: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Dark gripper fingers at bottom centre, as seen by the wrist camera (real frame: ~bottom quarter).
    Returns the image and the bool mask of gripper pixels (which hide whatever is under them)."""
    h, w = img.shape[:2]
    out, gm = img.copy(), np.zeros((h, w), np.uint8)
    top = int(h * rng.uniform(0.72, 0.8))
    for side in (-1, 1):
        cx = w / 2 + side * w * rng.uniform(0.02, 0.06)
        poly = np.array([[cx - w * 0.06, h], [cx + w * 0.06, h], [cx + w * 0.03, top], [cx - w * 0.03, top]], np.int32)
        cv2.fillPoly(out, [poly], tuple(int(c) for c in rng.uniform(20, 55, 3)))
        cv2.fillPoly(gm, [poly], 1)
    return out, gm.astype(bool)
