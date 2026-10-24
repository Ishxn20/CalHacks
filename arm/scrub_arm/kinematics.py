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
    def _target_rotation(self, yaw: float) -> np.ndarray:
        """World rotation that maps tool_axis_local -> -z and jaw_axis_local -> (cos yaw, sin yaw, 0)."""
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

    def tilt_from_down(self, q: np.ndarray) -> float:
        """Angle (rad) between where the fingers point and straight down."""
        _, R = self.fk(q)
        return float(np.arccos(np.clip(-(R @ self.tool_axis_local)[2], -1.0, 1.0)))

    def pivot_down(self, q_now: np.ndarray, limits: tuple[np.ndarray, np.ndarray], yaw: float | None = None,
                   step_deg: float = 2.0, max_joint_step_deg: float = 15.0) -> tuple[np.ndarray, float]:
        """Joint waypoints that turn the gripper to point straight down with its tip as the pivot.

        The orientation is interpolated along the shortest rotation in `step_deg` slices and each slice is
        solved with the tip pinned at its current position, so the tip stays put for the whole motion,
        not just at the ends. yaw: base-frame direction the jaws should close along at the end; None keeps
        the current jaw heading. Either way the symmetric flip (yaw + pi) is used if it's the shorter turn.
        Returns (waypoints, final yaw). Raises ValueError if the tip can't be held still the whole way.
        """
        pin = self.pin
        q_now = np.asarray(q_now, dtype=float)
        p, R0 = self.fk(q_now)
        if yaw is None:
            jaw = R0 @ self.jaw_axis_local
            if np.hypot(jaw[0], jaw[1]) < 0.2:       # jaws nearly vertical: no heading to keep
                yaw = float(np.arctan2(p[1], p[0]) + np.pi / 2)
            else:
                yaw = float(np.arctan2(jaw[1], jaw[0]))
        yaw = min((yaw, yaw + np.pi), key=lambda a: np.linalg.norm(pin.log3(R0.T @ self._target_rotation(a))))
        w = pin.log3(R0.T @ self._target_rotation(yaw))   # the turn, as a rotation vector in the tool frame
        n = max(1, int(np.ceil(np.degrees(np.linalg.norm(w)) / step_deg)))
        qs = [q_now]
        for k in range(1, n + 1):
            target = pin.SE3(R0 @ pin.exp3(w * k / n), p)
            q = self._solve(target, qs[-1], limits)
            if q is None:
                raise ValueError(f"can't keep the tip still past {np.degrees(np.linalg.norm(w)) * (k - 1) / n:.0f} "
                                 f"of {np.degrees(np.linalg.norm(w)):.0f} deg of turn (joint limit or out of reach); "
                                 "move the tip closer to the table / the middle of the workspace")
            jump = np.degrees(np.max(np.abs(q - qs[-1])))
            if jump > max_joint_step_deg:
                raise ValueError(f"turning down needs a {jump:.0f} deg joint jump here (near a singularity)")
            qs.append(q)
        return np.array(qs), float(np.arctan2(np.sin(yaw), np.cos(yaw)))

    def ik_topdown(self, x: float, y: float, z: float, yaw: float, q_seed: np.ndarray,
                   limits: tuple[np.ndarray, np.ndarray], allow_flip: bool = True) -> np.ndarray | None:
        """Joint vector placing the tool at (x, y, z) m pointing straight down, jaws along `yaw`; None if unreachable.

        allow_flip: a two-finger pinch is symmetric, so yaw + pi grips equally well. Pass False when the
        instrument's tip must end up on a known side of the gripper (the requested instrument: it is presented
        handle-first from a taught pose)."""
        q_seed = np.asarray(q_seed, dtype=float)
        # Damped least squares is local: also start from the base turned toward the target with the
        # wrist pre-rotated, which escapes the minima a far-away seed (e.g. the scan pose) gets stuck in.
        base = np.arctan2(y, x)
        seeds = [q_seed] + [np.array([base, np.radians(60), np.radians(70), 0.0, np.radians(50), w])
                            for w in (0.0, np.pi / 2, -np.pi / 2, 0.9 * np.pi, -0.9 * np.pi)]
        targets = [self.pin.SE3(self._target_rotation(yy), np.array([x, y, z], dtype=float))
                   for yy in ((yaw, yaw + np.pi) if allow_flip else (yaw,))]
        for group in (seeds[:1], seeds[1:]):       # restarts only if the direct solve fails both ways
            sols = [q for t in targets for s in group if (q := self._solve(t, s, limits)) is not None]
            if sols:
                return min(sols, key=lambda q: np.linalg.norm(q - q_seed))
        return None
