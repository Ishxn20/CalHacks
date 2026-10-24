"""Score a camera-guided reach policy (sim/servo_env.py) three ways, 50 episodes each by default:

  projected  the training observation (projected features + noise)
  nudged     the same, but the instrument is pushed 5-20 mm partway through the reach
  perceived  the wrist camera is RENDERED every step (MuJoCo, the real arm + instrument meshes) and the
             target's exact instance mask from segmentation rendering goes through scrub_vision identify()
             with the camera's current pixel->table homography; the policy sees features of THAT estimate.
             Up close the wrist camera rarely sees a whole instrument; identify() is trusted only when it
             measures the scanned length (+-8 %), otherwise the visible shank is tracked instead: its PCA axis, with the grasp point
             moved onto that line from the last estimate (initially the scan's full top-down view).

    python -m sim.eval_servo --policy sim/policy/reach_V1_best.zip [--n 50]
Success = true tool pose within 3 mm / 3 deg at the last step.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "arm"), str(ROOT / "perception")]

import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402
from scrub_vision.identify import _apply, identify, measure  # noqa: E402
from scrub_vision.render import scene_mesh  # noqa: E402

from sim.arm_model import XML, ARM_JOINTS  # noqa: E402
from sim.declutter_eval import _assets  # noqa: E402
from sim.servo_env import VisualReachEnv, evaluate, make_target  # noqa: E402
from sim.wrist_cam import CAM_H, CAM_W, F_PX, WristCam  # noqa: E402


def world_poses(t: dict) -> list[tuple[str, np.ndarray]]:
    """Every instrument of the target's pile, moved like the target (servo_env.placed), as 4x4 metre poses."""
    c, s = np.cos(t["phi"]), np.sin(t["phi"])
    M = np.eye(4)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[:2, 3] = t["at"] - M[:2, :2] @ (t["grasp_pile"])
    out = []
    for fid, T in t["pile"]:
        Tm = T.copy()
        Tm[:3, 3] /= 1000.0
        out.append((fid, M @ Tm))
    return out


class Perceiver:
    """Renders the wrist camera on a model with the arm and the episode's instruments; returns identify()'s
    estimate of the target (the pile's last instrument) or None when too little of it is visible."""

    def __init__(self, t: dict) -> None:
        insts, self.templates = _assets()
        self.insts = insts
        spec = mujoco.MjSpec.from_file(str(XML))
        self.target_geoms = []
        for k, (fid, T) in enumerate(world_poses(t)):
            m = scene_mesh(insts[fid])
            spec.add_mesh(name=f"inst{k}", uservert=(m.vertices / 1000.0).ravel().tolist(),
                          userface=np.r_[m.faces, m.faces[:, ::-1]].ravel().tolist())   # double-sided
            q = np.zeros(4)
            mujoco.mju_mat2Quat(q, T[:3, :3].ravel())
            spec.worldbody.add_geom(name=f"inst{k}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"inst{k}",
                                    pos=T[:3, 3], quat=q, contype=0, conaffinity=0, rgba=[0.5, 0.2, 0.7, 1])
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.tid = self.m.geom(f"inst{len(t['pile']) - 1}").id
        self.cam = WristCam(self.m)
        self.r = mujoco.Renderer(self.m, CAM_H, CAM_W)
        self.r.enable_segmentation_rendering()
        self.adr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in ARM_JOINTS])
        self.z = t["z"]
        self.last = {k: t[k] for k in ("grasp", "centroid", "axis", "length", "z")}   # the scan's estimate
        self.n_identify = self.n_track = 0
        self.scan_length = t["length"]       # a shorter "instrument" is this one cut off by the frame or fingers

    def __call__(self, env: VisualReachEnv) -> dict:
        self.m.cam_pos[self.cam.id] = env.model.cam_pos[env.cam.id]
        self.m.cam_quat[self.cam.id] = env.model.cam_quat[env.cam.id]
        self.d.qpos[self.adr] = env.q
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_camlight(self.m, self.d)
        self.r.update_scene(self.d, camera="wrist")
        seg = self.r.render()
        mask = (seg[..., 0] == self.tid) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM)
        if mask.sum() < 300:
            return {"hidden": True}
        H = self.homography()
        det = identify(mask, None, H, self.templates, self.insts)
        if det.candidates and abs(det.length_mm / 1000 - self.scan_length) < 0.08 * self.scan_length:
            self.n_identify += 1
            self.last = {"grasp": np.array(det.grasp_mm) / 1000, "centroid": _apply(H, det.centroid_px)[0] / 1000,
                         "axis": np.radians(det.axis_deg), "length": det.length_mm / 1000, "z": self.z}
            return self.last
        self.n_track += 1                                       # truncated view: track the shank line
        _, _, axis_deg, cpx = measure(mask, H)
        c, a = _apply(H, cpx)[0] / 1000, np.radians(axis_deg)
        if np.cos(a - self.last["axis"]) < 0:
            a += np.pi                                          # keep the tip direction
        u = np.array([np.cos(a), np.sin(a)])
        g = c + u * np.dot(self.last["grasp"] - c, u)
        self.last = {**self.last, "grasp": g, "centroid": self.last["centroid"] + g - self.last["grasp"], "axis": a}
        return self.last

    def homography(self) -> np.ndarray:
        """Pixel -> table-plane mm (at the instrument's mid height) for the current camera pose."""
        R = self.d.cam_xmat[self.cam.id].reshape(3, 3)
        c = self.d.cam_xpos[self.cam.id]
        mm = []
        for u, v in [(0, 0), (CAM_W, 0), (CAM_W, CAM_H), (0, CAM_H)]:
            ray = R @ np.array([(u - CAM_W / 2), -(v - CAM_H / 2), -F_PX])
            s = (self.z - c[2]) / ray[2]
            mm.append((c + s * ray)[:2] * 1000)
        px = np.float32([(0, 0), (CAM_W, 0), (CAM_W, CAM_H), (0, CAM_H)])
        return cv2.getPerspectiveTransform(px, np.float32(mm)).astype(np.float64)

    def close(self) -> None:
        self.r.close()


def perceived(model, n: int, seed: int) -> dict:
    env = VisualReachEnv(seed=seed, noise=False)          # perception supplies the noise now
    rng = np.random.default_rng(seed)
    ok, pe, re_, seen = [], [], [], []
    for i in range(n):
        t = make_target(rng)
        obs, _ = env.reset(seed=seed + 1 + i, options={"target": t})
        p = Perceiver(env.target)
        obs = env._obs(p(env))
        vis = []
        for _ in range(env.max_steps):
            obs, _, _, _, info = env.step(model.predict(obs, deterministic=True)[0], perceive=p)
            vis.append(obs[0])
        p.close()
        ok.append(info["success"])
        pe.append(info["pos_err"] * 1000)
        re_.append(info["rot_err_deg"])
        seen.append(float(np.mean(vis)))
        print(f"{i + 1}/{n} {'ok ' if info['success'] else 'bad'} {pe[-1]:.1f} mm {re_[-1]:.1f} deg "
              f"visible {seen[-1]:.0%}", flush=True)
    return {"success": float(np.mean(ok)), "pos_mm_median": float(np.median(pe)),
            "pos_mm_p95": float(np.percentile(pe, 95)), "rot_deg_median": float(np.median(re_)),
            "visible_frac": float(np.mean(seen))}


def main(argv=None) -> None:
    from stable_baselines3 import SAC

    ap = argparse.ArgumentParser(prog="sim.eval_servo")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--out", default="sim/policy/servo_eval.json")
    ap.add_argument("--skip-perceived", action="store_true")
    args = ap.parse_args(argv)
    model = SAC.load(args.policy, device="cpu")
    res = {"policy": args.policy,
           "projected": evaluate(model, args.n, args.seed),
           "nudged": evaluate(model, args.n, args.seed, nudge=True)}
    print(json.dumps(res, indent=1), flush=True)
    if not args.skip_perceived:
        res["perceived"] = perceived(model, args.n, args.seed)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
