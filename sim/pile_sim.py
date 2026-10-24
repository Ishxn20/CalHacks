"""MuJoCo pile generator: drop instruments onto a table, let physics settle them, return their poses.

Poses are rendered by perception's own rasteriser (scrub_vision.render), so the MuJoCo track and the
random-placement track produce identical label formats; no MuJoCo OpenGL needed on macOS.

    python -m scrub_vision.synth --source mujoco --n 3000 --out data/synth_mj   (from perception/)
"""
from __future__ import annotations

import numpy as np

from scrub_vision.render import scene_mesh
from scrub_vision.synth import PILE_CENTER_MM, PILE_HALF_MM, Pile
from scrub_vision.templates import load_instruments

DENSITY_PLA = 1240.0            # kg/m^3
_insts = None


def _instruments():
    global _insts
    if _insts is None:
        _insts = load_instruments()
    return _insts


def _quat(yaw: float, flip: bool) -> np.ndarray:
    """wxyz quaternion for flip (pi about x) then yaw about z."""
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    if not flip:
        return qz
    qx = np.array([0.0, 1.0, 0.0, 0.0])
    w1, x1, y1, z1 = qz
    w2, x2, y2, z2 = qx
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def _mjcf(items: list[str]) -> str:
    insts = _instruments()
    assets, bodies = [], []
    for i, fid in enumerate(items):
        m = scene_mesh(insts[fid])
        hull = m.convex_hull
        v = " ".join(f"{x:.6f}" for x in (hull.vertices / 1000.0).ravel())
        assets.append(f'<mesh name="m{i}" vertex="{v}"/>')
        bodies.append(f'<body name="b{i}"><joint type="free" damping="0.0002"/><geom type="mesh" mesh="m{i}" density="{DENSITY_PLA}" '
                      f'friction="0.6 0.005 0.0001" condim="4"/></body>')
    return f"""<mujoco><option timestep="0.001" integrator="implicitfast" cone="elliptic" impratio="10" gravity="0 0 -9.81"/>
<asset>{''.join(assets)}</asset>
<worldbody><geom type="plane" size="1 1 0.01" friction="0.7 0.005 0.0001"/>{''.join(bodies)}</worldbody></mujoco>"""


def settle_pile(items: list[str], rng: np.random.Generator, max_steps: int = 6000) -> tuple[Pile, float]:
    """Drop `items` one above another over the pile zone.

    Returns ([(fine_id, 4x4 mm)], drift): drift = largest body movement (mm, plus mm-equivalent of
    rotation over a 100 mm lever) over the last 200 steps. Thin flat hulls chatter a little in contact,
    so "settled" means the pose stopped changing, not that velocity is exactly zero.
    """
    import mujoco

    model = mujoco.MjModel.from_xml_string(_mjcf(items))
    data = mujoco.MjData(model)
    cx, cy = PILE_CENTER_MM
    hx, hy = PILE_HALF_MM
    for i in range(len(items)):
        adr = model.jnt_qposadr[model.body_jntadr[model.body(f"b{i}").id]]
        x = rng.uniform(cx - hx * 0.7, cx + hx * 0.7) / 1000
        y = rng.uniform(cy - hy * 0.7, cy + hy * 0.7) / 1000
        data.qpos[adr:adr + 3] = (x, y, 0.02 + 0.015 * i)
        data.qpos[adr + 3:adr + 7] = _quat(rng.uniform(-np.pi, np.pi), bool(rng.random() < 0.5))
    def snap():
        return np.array([np.r_[data.body(f"b{i}").xpos * 1000, data.body(f"b{i}").xmat * 100]
                         for i in range(len(items))])

    prev, drift = snap(), np.inf
    for step in range(1, max_steps + 1):
        mujoco.mj_step(model, data)
        if step % 200 == 0:
            cur = snap()
            drift = float(np.abs(cur - prev).max())
            prev = cur
            if step >= 1000 and drift < 0.5:
                break
    pile: Pile = []
    for i, fid in enumerate(items):
        b = data.body(f"b{i}")
        T = np.eye(4)
        T[:3, :3] = b.xmat.reshape(3, 3)
        T[:3, 3] = b.xpos * 1000.0
        pile.append((fid, T))
    return pile, drift


def poses_source(rng: np.random.Generator) -> Pile:
    insts = list(_instruments())
    n = int(rng.integers(3, 9))
    items = [str(f) for f in rng.choice(insts, size=n, replace=False)]
    return settle_pile(items, rng)[0]
