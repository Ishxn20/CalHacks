"""Watch the declutter loop live: MuJoCo physics (left) + what the wrist camera sees and decides (right).

    .venv/bin/python -m sim.watch                      # live window + sim/watch.mp4
    .venv/bin/python -m sim.watch --episodes 3 --no-window --out demo.mp4
Each episode: instruments drop into a pile and settle; the planner (exact instance masks, as in the stress
test) picks what to move; the chosen instrument lifts away to the staging row and the pile re-settles,
until the requested instrument is uncovered and lifted. The arm itself is not simulated yet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "perception"))

from scrub_vision.identify import annotate, identify  # noqa: E402
from scrub_vision.planner import display_name, plan  # noqa: E402
from scrub_vision.render import CamSpec, random_filament_bgr, render_scene, scene_mesh  # noqa: E402
from scrub_vision.synth import PILE_CENTER_MM, PILE_HALF_MM  # noqa: E402
from scrub_vision.templates import build_templates, interchangeable, load_instruments  # noqa: E402

from sim.pile_sim import _mjcf, _quat  # noqa: E402

W, H_PX = 960, 540
FPS = 30
CAM = CamSpec()
S = CAM.height_mm / CAM.fx
H = np.array([[S, 0, -CAM.w / 2 * S], [0, S, -CAM.h / 2 * S], [0, 0, 1]])
STAGING = [(320.0, -150.0 + 75 * i) for i in range(6)]     # a row beside the pile, outside the camera view


class Episode:
    def __init__(self, rng, insts, templates, items, target, show, writer):
        import mujoco

        self.mj, self.rng, self.insts, self.templates = mujoco, rng, insts, templates
        self.items, self.target, self.show, self.writer = items, target, show, writer
        set_colour = random_filament_bgr(rng)            # one filament for the whole set
        self.colours = [np.clip(set_colour * rng.uniform(0.93, 1.07), 0, 255) for _ in items]
        xml = _mjcf(items)
        rgba = "".join(f'<material name="c{i}" rgba="{c[2]/255:.3f} {c[1]/255:.3f} {c[0]/255:.3f} 1"/>'
                       for i, c in enumerate(self.colours))
        vis_assets = []
        for i, fid in enumerate(items):
            mesh = scene_mesh(insts[fid])
            v = " ".join(f"{x:.6f}" for x in (mesh.vertices / 1000.0).ravel())
            f = " ".join(str(int(x)) for x in mesh.faces.ravel())
            vis_assets.append(f'<mesh name="v{i}" vertex="{v}" face="{f}"/>')
            # collide with the hidden convex hull (group 3), draw the real mesh
            xml = xml.replace(f'<geom type="mesh" mesh="m{i}" ',
                              f'<geom type="mesh" mesh="v{i}" material="c{i}" contype="0" conaffinity="0" '
                              f'density="0"/><geom group="3" type="mesh" mesh="m{i}" ', 1)
        xml = xml.replace("<asset>", "<asset>" + "".join(vis_assets), 1)
        xml = xml.replace("<asset>", f"<asset>{rgba}"
                          '<texture name="t" type="2d" builtin="checker" rgb1=".62 .62 .64" rgb2=".58 .58 .6" width="64" height="64"/>'
                          '<material name="table" texture="t" texrepeat="20 20"/>')
        xml = xml.replace('<geom type="plane"', '<geom type="plane" material="table"')
        xml = xml.replace("<worldbody>", f'<visual><global offwidth="{W}" offheight="{H_PX}"/></visual>'
                          '<worldbody><light pos="0 -0.3 1.2" dir="0 0.2 -1" diffuse=".8 .8 .8"/>'
                          '<light pos="0.5 0.5 1" dir="-0.4 -0.4 -1" diffuse=".4 .4 .4"/>')
        self.m = mujoco.MjModel.from_xml_string(xml)
        self.d = mujoco.MjData(self.m)
        self.r = mujoco.Renderer(self.m, H_PX, W)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = (0.10, -0.01, 0.0)
        self.cam.distance, self.cam.elevation, self.cam.azimuth = 0.62, -52, 115
        self.adr = [self.m.jnt_qposadr[self.m.body_jntadr[self.m.body(f"b{i}").id]] for i in range(len(items))]
        self.vadr = [self.m.jnt_dofadr[self.m.body_jntadr[self.m.body(f"b{i}").id]] for i in range(len(items))]
        self.staged: set[int] = set()
        self.right = np.full((H_PX, W, 3), 30, np.uint8)
        cv2.putText(self.right, "waiting for the pile to settle...", (W // 2 - 220, H_PX // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        self.caption = f"Request: {display_name(target)}"

    # ── drawing ───────────────────────────────────────────────────────────
    def frame(self) -> bool:
        self.r.update_scene(self.d, self.cam)
        left = cv2.cvtColor(self.r.render(), cv2.COLOR_RGB2BGR)
        out = np.hstack([left, self.right])
        cv2.rectangle(out, (0, 0), (2 * W, 44), (25, 25, 25), -1)
        cv2.putText(out, self.caption, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        cv2.putText(out, "physics (MuJoCo)", (14, H_PX - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, "wrist camera + decision", (W + 14, H_PX - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 255), 1)
        if self.writer is not None:
            self.writer.write(out)
        if self.show:
            cv2.imshow("scrub arm: declutter simulation", out)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                return False
        return True

    def run_physics(self, seconds: float, hold: dict | None = None) -> bool:
        steps = int(1.0 / FPS / self.m.opt.timestep)
        for _ in range(int(seconds * FPS)):
            for _ in range(steps):
                if hold:
                    for i, q in hold.items():
                        self.d.qpos[self.adr[i]:self.adr[i] + 7] = q
                        self.d.qvel[self.vadr[i]:self.vadr[i] + 6] = 0
                self.mj.mj_step(self.m, self.d)
            if not self.frame():
                return False
        return True

    # ── perception ────────────────────────────────────────────────────────
    def pile(self):
        out = []
        for i, fid in enumerate(self.items):
            if i in self.staged:
                continue
            b = self.d.body(f"b{i}")
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = b.xmat.reshape(3, 3), b.xpos * 1000
            out.append((i, fid, T))
        return out

    def look(self):
        pile = self.pile()
        bgr, masks = render_scene([(scene_mesh(self.insts[f]), T) for _, f, T in pile], CAM, self.rng,
                                  colours=[self.colours[i] for i, _, _ in pile])
        dets, truth = [], {}
        for (i, _, _), mk in zip(pile, masks):
            if mk.sum() >= 300:
                d = identify(mk, None, H, self.templates, self.insts)
                dets.append(d)
                truth[id(d)] = i
        return bgr, dets, truth

    # ── motion of a picked instrument (kinematic, like the gripper carrying it) ──
    def carry(self, i: int, to_xy_mm: tuple[float, float] | None, seconds: float = 2.0) -> bool:
        a = self.adr[i]
        p0, q0 = self.d.qpos[a:a + 3].copy(), self.d.qpos[a + 3:a + 7].copy()
        up = p0 + [0, 0, 0.10]
        n = int(seconds * FPS)
        if to_xy_mm is None:                       # requested instrument: lift and hold up
            path = [p0 + (up - p0) * t for t in np.linspace(0, 1, n)]
        else:
            dest = np.array([to_xy_mm[0] / 1000, to_xy_mm[1] / 1000, 0.10])
            down = dest.copy()
            down[2] = scene_mesh(self.insts[self.items[i]]).extents[2] / 2000 + 0.002
            k = n // 3
            path = ([p0 + (up - p0) * t for t in np.linspace(0, 1, k)] +
                    [up + (dest - up) * t for t in np.linspace(0, 1, k)] +
                    [dest + (down - dest) * t for t in np.linspace(0, 1, n - 2 * k)])
        flat = _quat(0.0, False)
        for j, p in enumerate(path):
            q = q0 if (to_xy_mm is None or j < len(path) * 2 // 3) else flat
            if not self.run_physics(1.0 / FPS, hold={i: np.r_[p, q]}):
                return False
        return True

    def play(self, max_moves: int = 6) -> str:
        cx, cy = PILE_CENTER_MM
        hx, hy = PILE_HALF_MM
        for i in range(len(self.items)):
            a = self.adr[i]
            self.d.qpos[a:a + 3] = (self.rng.uniform(cx - hx * .7, cx + hx * .7) / 1000,
                                    self.rng.uniform(cy - hy * .7, cy + hy * .7) / 1000, 0.03 + 0.02 * i)
            self.d.qpos[a + 3:a + 7] = _quat(self.rng.uniform(-np.pi, np.pi), bool(self.rng.random() < .5))
        self.caption = f"Request: {display_name(self.target)}  -  dropping {len(self.items)} instruments"
        if not self.run_physics(2.5):
            return "quit"
        used, moves = set(), 0
        while True:
            bgr, dets, truth = self.look()
            p = plan(dets, self.target, H, STAGING, used)
            chosen = None
            if p.get("target"):
                chosen = min(dets, key=lambda d: np.hypot(d.grasp_mm[0] - p["target"]["x_mm"],
                                                          d.grasp_mm[1] - p["target"]["y_mm"]))
            view = annotate(bgr, dets, chosen)
            self.right = cv2.resize(view, (W, H_PX))
            tag = {"grasp_target": "GRASP TARGET", "move_aside": "MOVE ASIDE", "fail": "GIVE UP"}[p["action"]]
            self.caption = f"Request: {display_name(self.target)}  -  {tag}: {p['say']}"
            if not self.run_physics(1.2):
                return "quit"
            if p["action"] == "fail":
                return "fail"
            i = truth[id(chosen)]
            if p["action"] == "grasp_target":
                ok = self.items[i] in interchangeable(self.insts, self.target)
                self.caption = (f"Request: {display_name(self.target)}  -  "
                                f"{'GOT IT' if ok else 'WRONG: ' + display_name(self.items[i])} after {moves} move(s)")
                self.carry(i, None)
                self.run_physics(1.0, hold={i: self.d.qpos[self.adr[i]:self.adr[i] + 7].copy()})
                return "grasped" if ok else "wrong"
            used.add(p["staging_index"])
            self.carry(i, STAGING[p["staging_index"]])
            self.staged.add(i)
            if not self.run_physics(1.0):
                return "quit"
            moves += 1
            if moves >= max_moves:
                return "cap"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="sim.watch")
    ap.add_argument("--episodes", type=int, default=5)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="sim/watch.mp4")
    ap.add_argument("--no-window", action="store_true")
    args = ap.parse_args(argv)
    insts = load_instruments()
    templates = build_templates(insts, 0.5)
    rng = np.random.default_rng(args.seed)
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"avc1"), FPS, (2 * W, H_PX)) if args.out else None
    results = []
    for _ in range(args.episodes):
        items = [str(f) for f in rng.choice(list(insts), size=int(rng.integers(5, 9)), replace=False)]
        target = str(rng.choice(items))
        res = Episode(rng, insts, templates, items, target, not args.no_window, writer).play()
        results.append(res)
        print(f"{display_name(target)}: {res}", flush=True)
        if res == "quit":
            break
    if writer is not None:
        writer.release()
        print(f"video: {args.out}")
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
