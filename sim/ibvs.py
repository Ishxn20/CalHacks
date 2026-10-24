"""Image-based visual servo for sim/servo_env.py, from the camera features + the arm's own joint angles only.

Mount: the gripper landmarks' pixels (servo_env.LANDMARKS) depend only on the camera mount, so their running
mean over the episode is fitted (Gauss-Newton on back, side, tilt) -> an online hand-eye calibration.
Each step: the target pixel is back-projected through the fitted camera onto the table plane -> grasp point
estimate; the
image angle between target axis and jaws gives the jaw yaw (either way round, whichever is nearer). The tool
then takes a damped-least-squares step toward (estimate, table + h) with a gain, through the joint Jacobian.
Estimates are smoothed over frames (moving average, alpha; yaw averaged as a doubled angle) since the target
is still between nudges; while it is hidden the last estimate is kept, and before it has ever been seen the
tool backs up away from the table to widen the view. Output: the env's action (cube root of dq / MAX_DQ).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics
from sim.servo_env import LANDMARKS, MAX_DQ, N_OBS, VisualReachEnv
from sim.wrist_cam import CAM_H, CAM_W, F_PX, NOMINAL, mount_pose

_PLANE_Z = 0.002          # instruments are ~3-9 mm thick: aim at ~their mid height


def _quat_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class IBVS:
    def __init__(self, gain: float = 0.5, damping: float = 1e-4, alpha: float = 0.2) -> None:
        cfg = load_arm_config()
        self.kin, self.table_z = Kinematics(), cfg.pick.table_z
        self.gain, self.damping, self.alpha = gain, damping, alpha
        self.reset()

    def reset(self) -> None:
        self.est, self.lm_sum, self.lm_n, self.backed = None, np.zeros(6), 0, 0
        self._set_mount(dict(NOMINAL))

    def _set_mount(self, mount: dict) -> None:
        self.mount = mount
        pos, quat = mount_pose(**mount)
        self.cam_p, self.cam_R = pos, _quat_mat(quat)          # in the tool (gripper_end) frame

    def _landmark_px(self, mount: dict) -> np.ndarray:
        pos, quat = mount_pose(**mount)
        return np.concatenate([self._project(pos, _quat_mat(quat), L) for L in LANDMARKS])

    def fit_mount(self, obs: np.ndarray) -> None:
        """Running mean of the landmark pixels -> mount, by Gauss-Newton from the current fit."""
        self.lm_sum += obs[21:27] * np.tile([CAM_W, CAM_H], 3)
        self.lm_n += 1
        seen, x = self.lm_sum / self.lm_n, np.array([self.mount["back"], self.mount["side"], self.mount["tilt_deg"]])
        as_mount = lambda x: {"back": x[0], "side": x[1], "tilt_deg": x[2]}  # noqa: E731
        for _ in range(5):
            r = self._landmark_px(as_mount(x)) - seen
            J = np.column_stack([(self._landmark_px(as_mount(x + dx)) - self._landmark_px(as_mount(x - dx))) / (2 * dx[i])
                                 for i, dx in enumerate(np.diag([1e-4, 1e-4, 1e-2]))])
            x = x - np.linalg.lstsq(J, r, rcond=None)[0]
        self._set_mount(as_mount(x))

    # ── camera model ──────────────────────────────────────────────────────
    def _camera(self, q: np.ndarray):
        p, R = self.kin.fk(q)
        return p + R @ self.cam_p, R @ self.cam_R

    @staticmethod
    def _project(c, Rc, X) -> np.ndarray:
        pc = (X - c) @ Rc
        return np.array([CAM_W / 2 + F_PX * pc[0] / -pc[2], CAM_H / 2 - F_PX * pc[1] / -pc[2]])

    def _ray_to_plane(self, c, Rc, uv) -> np.ndarray:
        ray = Rc @ np.array([uv[0] - CAM_W / 2, -(uv[1] - CAM_H / 2), -F_PX])
        return (c + (self.table_z + _PLANE_Z - c[2]) / ray[2] * ray)[:2]

    def estimate(self, obs: np.ndarray) -> dict | None:
        """Grasp point (base-frame xy) and jaw yaw from the features, or None if the target isn't visible."""
        if obs[0] < 0.5:
            return None
        q = obs[8:14] * np.pi
        c, Rc = self._camera(q)
        p, R = self.kin.fk(q)
        tip = self._project(c, Rc, p)                                  # where the fingertips appear (fitted mount)
        g = tip + obs[1:3] * [CAM_W, CAM_H]                            # the target, relative to them
        jaw = R @ self.kin.jaw_axis_local
        dj = self._project(c, Rc, p + 0.02 * jaw) - tip
        rel = np.arctan2(obs[3], obs[4]) / 2                           # target axis - jaws, in the image (mod pi)
        a = np.arctan2(dj[1], dj[0]) + rel
        xy = self._ray_to_plane(c, Rc, g)
        xy2 = self._ray_to_plane(c, Rc, g + 40 * np.array([np.cos(a), np.sin(a)]))
        axis = np.arctan2(*(xy2 - xy)[::-1])
        return {"xy": xy, "yaw": axis + np.pi / 2}

    # ── control ───────────────────────────────────────────────────────────
    def dq(self, obs: np.ndarray) -> np.ndarray:
        self.fit_mount(obs)
        est = self.estimate(obs)
        if est is not None:
            if self.est is None:
                self.est = est
            else:
                a = self.alpha
                d2 = np.angle(np.exp(2j * est["yaw"]) / np.exp(2j * self.est["yaw"]))   # doubled-angle difference
                self.est = {"xy": (1 - a) * self.est["xy"] + a * est["xy"], "yaw": self.est["yaw"] + a * d2 / 2}
        q = obs[8:14] * np.pi
        pin = self.kin.pin
        if self.est is None:                                           # never seen: back up to widen the view
            self.backed += 1
            if self.backed > 8:                                        # 80 mm is enough; then hold still
                return np.zeros(6)
            p, R = self.kin.fk(q)
            target = pin.SE3(R, p - 0.01 * (R @ self.kin.tool_axis_local))
            return self._step_to(q, target)
        h = obs[20] * 0.1
        R_now = self.kin.fk(q)[1]
        yaws = [self.est["yaw"], self.est["yaw"] + np.pi]
        rot = [self.kin._target_rotation(y) for y in yaws]
        Rt = min(rot, key=lambda Rt: np.linalg.norm(pin.log3(R_now.T @ Rt)))      # nearer jaw direction
        return self._step_to(q, pin.SE3(Rt, np.array([*self.est["xy"], self.table_z + h])))

    def _step_to(self, q: np.ndarray, target) -> np.ndarray:
        pin = self.kin.pin
        qf = self.kin._pad(q)
        pin.forwardKinematics(self.kin.model, self.kin.data, qf)
        pin.updateFramePlacements(self.kin.model, self.kin.data)
        err = pin.log6(self.kin.data.oMf[self.kin.fid].inverse() * target).vector
        J = pin.computeFrameJacobian(self.kin.model, self.kin.data, qf, self.kin.fid, pin.LOCAL)[:, :6]
        dq = J.T @ np.linalg.solve(J @ J.T + self.damping * np.eye(6), err)
        return np.clip(self.gain * dq, -MAX_DQ, MAX_DQ)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return np.cbrt(self.dq(obs) / MAX_DQ)

    def predict(self, obs, deterministic=True):                         # stable-baselines3 interface
        return self(obs), None


RESIDUAL = 0.25          # learned correction: up to 25 % of MAX_DQ per joint per step


class ResidualEnv(VisualReachEnv):
    """VisualReachEnv where the agent's action is a bounded correction to the IBVS step. Observation: the
    camera observation + the IBVS step (dq / MAX_DQ). Zero correction = IBVS exactly."""

    def __init__(self, *args, **kw) -> None:
        super().__init__(*args, **kw)
        self.ctl = IBVS()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (N_OBS + 6,), np.float32)

    def _with_base(self, obs: np.ndarray) -> np.ndarray:
        self.base_dq = self.ctl.dq(obs)
        return np.concatenate([obs, self.base_dq / MAX_DQ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        self.ctl.reset()
        obs, info = super().reset(seed=seed, options=options)
        return self._with_base(obs), info

    def step(self, action, perceive=None):
        dq = self.base_dq + RESIDUAL * MAX_DQ * np.clip(np.asarray(action, float), -1, 1)
        obs, r, term, trunc, info = super().step(np.cbrt(np.clip(dq / MAX_DQ, -1, 1)), perceive)
        return self._with_base(obs), r, term, trunc, info
