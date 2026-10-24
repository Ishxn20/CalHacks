"""Forward / inverse kinematics for the B601-RS, straight from the URDF.

Plain numpy so it runs in any environment (Pinocchio, which Seeed's SDK uses,
needs Python < 3.12). The URDF is Seeed's ReBot_Arm_RS.urdf; motor angles map
1:1 onto its joint angles, the same assumption reBotArm_control_py makes.

Frames: base frame of the arm, metres. The tool point is the URDF's
`gripper_end` frame (between the jaws).
"""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_URDF = Path(__file__).resolve().parent.parent / "urdf" / "ReBot_Arm_RS.urdf"


class IKError(RuntimeError):
    pass


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    x, y, z = axis
    c, s, t = math.cos(angle), math.sin(angle), 1 - math.cos(angle)
    return np.array([
        [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
        [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
        [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
    ])


def rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis * angle."""
    cos = float(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if math.pi - angle < 1e-6:
        # 180 deg: axis from the largest diagonal term
        i = int(np.argmax(np.diag(R)))
        axis = R[:, i] + np.eye(3)[i]
        return axis / np.linalg.norm(axis) * angle
    w = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return w / (2 * math.sin(angle)) * angle


@dataclass
class _Joint:
    name: str
    origin: np.ndarray    # 4x4 parent -> joint frame
    axis: np.ndarray | None  # None for fixed joints
    child: str


class Kinematics:
    def __init__(self, joints: list[_Joint], lower: np.ndarray, upper: np.ndarray) -> None:
        self._joints = joints
        self.names = [j.name for j in joints if j.axis is not None]
        self.lower, self.upper = lower, upper

    @classmethod
    def from_urdf(cls, path: Path | str = DEFAULT_URDF, base: str = "base_link", tip: str = "gripper_end",
                  lower: np.ndarray | None = None, upper: np.ndarray | None = None) -> "Kinematics":
        root = ET.parse(path).getroot()
        by_child = {}
        for j in root.findall("joint"):
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            T = np.eye(4)
            T[:3, :3], T[:3, 3] = _rpy(*rpy), xyz
            axis = None
            if j.get("type") in ("revolute", "continuous"):
                a = np.array([float(v) for v in j.find("axis").get("xyz").split()])
                axis = a / np.linalg.norm(a)
            child = j.find("child").get("link")
            by_child[child] = (j.find("parent").get("link"), _Joint(j.get("name"), T, axis, child))
        chain, link = [], tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"no joint chain from {base} to {tip} in {path}")
            link, joint = by_child[link]
            chain.append(joint)
        chain.reverse()
        n = sum(j.axis is not None for j in chain)
        lo = np.full(n, -math.pi) if lower is None else np.asarray(lower, float)
        hi = np.full(n, math.pi) if upper is None else np.asarray(upper, float)
        return cls(chain, lo, hi)

    @property
    def n(self) -> int:
        return len(self.names)

    def frames(self, q: np.ndarray) -> list[tuple[np.ndarray, np.ndarray | None]]:
        """(world transform, world joint axis or None) after each joint in the chain."""
        T, out, i = np.eye(4), [], 0
        for j in self._joints:
            T = T @ j.origin
            if j.axis is not None:
                R = np.eye(4)
                R[:3, :3] = _axis_angle(j.axis, float(q[i]))
                axis_w = T[:3, :3] @ j.axis
                T = T @ R
                out.append((T.copy(), axis_w))
                i += 1
            else:
                out.append((T.copy(), None))
        return out

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.frames(q)[-1][0]

    def approach(self, q: np.ndarray) -> np.ndarray:
        """Unit vector the gripper points along: from the wrist (last joint) to the tool point."""
        frames = self.frames(q)
        wrist = [T for T, ax in frames if ax is not None][-1][:3, 3]
        d = frames[-1][0][:3, 3] - wrist
        return d / np.linalg.norm(d)

    def jacobian(self, q: np.ndarray) -> np.ndarray:
        frames = self.frames(q)
        p_tip = frames[-1][0][:3, 3]
        J = np.zeros((6, self.n))
        i = 0
        for T, axis in frames:
            if axis is None:
                continue
            J[:3, i] = np.cross(axis, p_tip - T[:3, 3])
            J[3:, i] = axis
            i += 1
        return J

    def ik(self, T_goal: np.ndarray, q0: np.ndarray, iters: int = 200,
           tol_pos: float = 1e-4, tol_rot: float = 1e-3) -> np.ndarray:
        """Damped least-squares IK for full pose, seeded from q0 and clamped to joint limits."""
        q = np.array(q0, dtype=float)
        lam = 0.01
        for _ in range(iters):
            T = self.fk(q)
            e = np.concatenate([T_goal[:3, 3] - T[:3, 3], rotvec(T_goal[:3, :3] @ T[:3, :3].T)])
            if np.linalg.norm(e[:3]) < tol_pos and np.linalg.norm(e[3:]) < tol_rot:
                return q
            J = self.jacobian(q)
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), e)
            step = np.max(np.abs(dq))
            if step > 0.2:  # keep each iteration small so we stay on the same branch
                dq *= 0.2 / step
            q = np.clip(q + dq, self.lower, self.upper)
        raise IKError(f"IK did not converge (pos err {np.linalg.norm(e[:3]) * 1000:.1f} mm, "
                      f"rot err {math.degrees(np.linalg.norm(e[3:])):.1f} deg)")

    def line(self, q0: np.ndarray, delta: np.ndarray, step: float = 0.002,
             max_joint_step: float = math.radians(5)) -> np.ndarray:
        """Joint waypoints that move the tool point by `delta` (m, base frame) in a straight line,
        keeping its orientation. Raises IKError if any point is unreachable or the arm would flip."""
        T0 = self.fk(q0)
        dist = float(np.linalg.norm(delta))
        n = max(1, math.ceil(dist / step))
        qs, q = [np.array(q0, dtype=float)], np.array(q0, dtype=float)
        for k in range(1, n + 1):
            T = T0.copy()
            T[:3, 3] = T0[:3, 3] + np.asarray(delta) * k / n
            q_next = self.ik(T, q)
            if np.max(np.abs(q_next - q)) > max_joint_step:
                raise IKError(f"path needs a {math.degrees(np.max(np.abs(q_next - q))):.0f} deg joint jump "
                              f"at {k * dist / n * 1000:.0f} mm (singularity or joint limit)")
            qs.append(q_next)
            q = q_next
        return np.array(qs)
