"""The wrist webcam in the MuJoCo arm model: mount pose (arm.yaml wrist_camera, randomisable) and pinhole
projection that matches MuJoCo's own rendering of the "wrist" camera.

Tool frame (gripper_end): +x = along the fingers (fingertips at the origin), y = jaw axis (-), z = across.
The camera sits `back` behind the fingertips and `side` off the axis in +z, its optical axis pitched
`tilt` toward the fingers; image right = jaw opening direction, fingertips at the bottom centre.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import yaml

_cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "arm" / "config" / "arm.yaml").read_text())["wrist_camera"]
CAM_W, CAM_H = int(_cfg["width"]), int(_cfg["height"])
HFOV = np.radians(float(_cfg["hfov_deg"]))
FOVY_DEG = float(np.degrees(2 * np.arctan(np.tan(HFOV / 2) * CAM_H / CAM_W)))
F_PX = (CAM_H / 2) / np.tan(np.radians(FOVY_DEG) / 2)
NOMINAL = {"back": _cfg["back_mm"] / 1000, "side": _cfg["side_mm"] / 1000, "tilt_deg": float(_cfg["tilt_deg"])}


def mount_pose(back: float, side: float, tilt_deg: float) -> tuple[np.ndarray, np.ndarray]:
    """(pos, wxyz quat) of the camera in the gripper_end frame."""
    t = np.radians(tilt_deg)
    up = np.array([np.sin(t), 0.0, np.cos(t)])          # camera +y (image up)
    z = -np.array([np.cos(t), 0.0, -np.sin(t)])         # camera looks along -z
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([np.cross(up, z), up, z]).ravel())
    return np.array([-back, 0.0, side]), q


class WristCam:
    def __init__(self, model: mujoco.MjModel, name: str = "wrist") -> None:
        self.id = model.camera(name).id

    def set_mount(self, model: mujoco.MjModel, back: float = NOMINAL["back"], side: float = NOMINAL["side"],
                  tilt_deg: float = NOMINAL["tilt_deg"]) -> None:
        model.cam_pos[self.id], model.cam_quat[self.id] = mount_pose(back, side, tilt_deg)

    def project(self, data: mujoco.MjData, pts: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixels (N, 2) (u right, v down); NaN for points behind the camera."""
        R = data.cam_xmat[self.id].reshape(3, 3)
        pc = (np.asarray(pts, float) - data.cam_xpos[self.id]) @ R
        depth = -pc[:, 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = np.column_stack([CAM_W / 2 + F_PX * pc[:, 0] / depth, CAM_H / 2 - F_PX * pc[:, 1] / depth])
        uv[depth <= 1e-6] = np.nan
        return uv
