"""FK / top-down IK on the RS URDF via pinocchio. Arm joints only (6); gripper joints held at 0."""
from __future__ import annotations

from pathlib import Path

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "urdf" / "ReBot_Arm_RS.urdf"
N_ARM = 6


class Kinematics:
    def __init__(self, urdf_path: Path | None = None, tool_frame: str = "gripper_end") -> None:
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path or DEFAULT_URDF))
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(tool_frame)
        self._link6 = self.model.getFrameId("link6")
        q0 = np.zeros(N_ARM)
        p6, R6 = self._fk_frame(q0, self._link6)
        pt, Rt = self._fk_frame(q0, self.fid)
        self.tool_axis_local = Rt.T @ (pt - p6) / np.linalg.norm(pt - p6)   # fingers point along this
        # Jaw axis: direction the left finger travels when gripper_joint1 opens, in tool frame.
        qa, qb = np.zeros(self.model.nq), np.zeros(self.model.nq)
        qb[N_ARM] = 0.01
        fl = self.model.getFrameId("gripper_left")
        d = self._fk_full(qb, fl)[0] - self._fk_full(qa, fl)[0]
        self.jaw_axis_local = Rt.T @ d / np.linalg.norm(d)
        self.tilt_R, self.tilt_yaw = None, 0.0     # set_pick_tilt: picks copy a taught pose's tilt

    def set_pick_tilt(self, q: np.ndarray) -> None:
        """Grasp poses copy this pose's tilt instead of pointing straight down, turned about the vertical to the
        requested jaw yaw, all the way down to the grab. The scan pose tilts the gripper 45 deg so the wrist camera
        looks straight down: a vertical descent then barely moves the target in the image (measured: within ~40 px
        from 10 cm to 3 cm), and it reaches ~100 mm further than straight down (the real mat sits at x 380-640 mm)."""
        _, R = self.fk(np.asarray(q, float))
        jaw = R @ self.jaw_axis_local
        self.tilt_R, self.tilt_yaw = R, float(np.arctan2(jaw[1], jaw[0]))

    # ── FK ────────────────────────────────────────────────────────────────
    def _pad(self, q: np.ndarray) -> np.ndarray:
        full = np.zeros(self.model.nq)
        full[:N_ARM] = q[:N_ARM]
        return full

    def _fk_full(self, qfull: np.ndarray, fid: int) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, qfull)
        self.pin.updateFramePlacements(self.model, self.data)
        M = self.data.oMf[fid]
        return M.translation.copy(), M.rotation.copy()

    def _fk_frame(self, q: np.ndarray, fid: int) -> tuple[np.ndarray, np.ndarray]:
        return self._fk_full(self._pad(q), fid)

    def fk(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tool frame position (m) and rotation in the base frame."""
        return self._fk_frame(np.asarray(q, dtype=float), self.fid)

    # ── IK ────────────────────────────────────────────────────────────────
    def _target_rotation(self, yaw: float, tilt: bool = True) -> np.ndarray:
        """World rotation that maps tool_axis_local -> -z and jaw_axis_local -> (cos yaw, sin yaw, 0); with a pick
        tilt, that pose's rotation turned about the vertical so its jaw axis lies along yaw."""
        if tilt and self.tilt_R is not None:
            c, s = np.cos(yaw - self.tilt_yaw), np.sin(yaw - self.tilt_yaw)
            return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ self.tilt_R
        t, j = self.tool_axis_local, self.jaw_axis_local
        j = j - np.dot(j, t) * t
        j /= np.linalg.norm(j)
        local = np.column_stack([t, j, np.cross(t, j)])
        tw = np.array([0.0, 0.0, -1.0])
        jw = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        world = np.column_stack([tw, jw, np.cross(tw, jw)])
        return world @ local.T

    def _solve(self, target, q_seed: np.ndarray, limits, max_iter=200, damping=1e-4, step=0.5):
        pin = self.pin
        lo, hi = limits
        q = self._pad(q_seed)
        for _ in range(max_iter):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            err = pin.log6(self.data.oMf[self.fid].inverse() * target).vector
            if np.linalg.norm(err) < 1e-5:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL)[:, :N_ARM]
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q[:N_ARM] = np.clip(q[:N_ARM] + step * dq, lo, hi)
        pos, _ = self._fk_full(q, self.fid)
        if np.linalg.norm(pos - target.translation) > 2e-3:
            return None
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        rot_err = pin.log3(self.data.oMf[self.fid].rotation.T @ target.rotation)
        if np.linalg.norm(rot_err) > np.radians(1.0):
            return None
        return q[:N_ARM].copy()

    def ik_pose(self, p: np.ndarray, R: np.ndarray, q_seed: np.ndarray,
                limits: tuple[np.ndarray, np.ndarray]) -> np.ndarray | None:
        """Joints placing the tool at position p (m) with rotation R, solved from q_seed only (no restarts), so
        consecutive small steps stay on one continuous solution (straight-line moves)."""
        return self._solve(self.pin.SE3(np.asarray(R, float), np.asarray(p, float)), np.asarray(q_seed, float), limits)

    def ik_topdown(self, x: float, y: float, z: float, yaw: float, q_seed: np.ndarray,
                   limits: tuple[np.ndarray, np.ndarray], allow_flip: bool = True,
                   tilt: bool | None = None) -> np.ndarray | None:
        """Joint vector placing the tool at (x, y, z) m pointing straight down (or with the pick tilt, see
        set_pick_tilt), jaws along `yaw`; None if unreachable. tilt: None = the pick tilt where it reaches, else
        straight down; True / False = only that.

        allow_flip: a two-finger pinch is symmetric, so yaw + pi grips equally well. Pass False when the
        instrument's tip must end up on a known side of the gripper (the requested instrument: it is presented
        handle-first from a taught pose)."""
        q_seed = np.asarray(q_seed, dtype=float)
        # Damped least squares is local: also start from the base turned toward the target with the
        # wrist pre-rotated, which escapes the minima a far-away seed (e.g. the scan pose) gets stuck in.
        base = np.arctan2(y, x)
        seeds = [q_seed] + [np.array([base, np.radians(60), np.radians(70), 0.0, np.radians(50), w])
                            for w in (0.0, np.pi / 2, -np.pi / 2, 0.9 * np.pi, -0.9 * np.pi)]
        # The pick tilt first; straight down where the tilt can't reach (near the base: the tilted wrist needs room)
        if self.tilt_R is None:
            tilt = False
        for tl in ((True, False) if tilt is None else (tilt,)):
            targets = [self.pin.SE3(self._target_rotation(yy, tl), np.array([x, y, z], dtype=float))
                       for yy in ((yaw, yaw + np.pi) if allow_flip else (yaw,))]
            for group in (seeds[:1], seeds[1:]):       # restarts only if the direct solve fails both ways
                sols = [q for t in targets for s in group if (q := self._solve(t, s, limits)) is not None]
                if sols:
                    return min(sols, key=lambda q: np.linalg.norm(q - q_seed))
        return None
