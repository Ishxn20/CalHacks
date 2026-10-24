import mujoco
import numpy as np
import pytest

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics
from sim.arm_model import XML, load_model, set_arm
from sim.reach_env import SCAN
from sim.wrist_cam import CAM_W, CAM_H, WristCam

P = np.array([0.28, 0.05, 0.002])        # a point on the table


@pytest.fixture(scope="module")
def hover():
    """Model + data with the tool 80 mm straight above P, jaws at 30 deg."""
    cfg, kin = load_arm_config(), Kinematics()
    q = kin.ik_topdown(P[0], P[1], P[2] + 0.08, np.radians(30), SCAN, (cfg.lower, cfg.upper), allow_flip=False)
    m = load_model()
    d = mujoco.MjData(m)
    set_arm(m, d, q)
    return m, d


def test_point_below_the_gripper_projects_above_the_fingers_at_bottom_centre(hover):
    m, d = hover
    cam = WristCam(m)
    u, v = cam.project(d, P[None])[0]
    tu, tv = cam.project(d, d.site("tool").xpos[None])[0]
    assert abs(u - CAM_W / 2) < 0.05 * CAM_W and 0.5 * CAM_H < v < 0.75 * CAM_H
    assert abs(tu - CAM_W / 2) < 0.05 * CAM_W and tv > 0.75 * CAM_H        # fingertips: bottom quarter


def test_mount_offset_moves_the_projection(hover):
    m, d = hover
    cam = WristCam(m)
    base = cam.project(d, P[None])[0]
    cam.set_mount(m, back=0.120, side=0.060, tilt_deg=13.0)
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    moved = cam.project(d, P[None])[0]
    cam.set_mount(m)                          # back to nominal
    mujoco.mj_kinematics(m, d)
    mujoco.mj_camlight(m, d)
    assert np.linalg.norm(moved - base) > 10
    assert np.allclose(cam.project(d, P[None])[0], base)


def test_projection_matches_the_mujoco_render(hover):
    m, d = hover
    xml = XML.read_text().replace('meshdir="shared"', f'meshdir="{XML.parent / "shared"}"')
    xml = xml.replace("<worldbody>", f'<worldbody><geom name="marker" type="sphere" size="0.003" '
                      f'pos="{P[0]} {P[1]} {P[2]}" contype="0" conaffinity="0"/>', 1)
    m2 = mujoco.MjModel.from_xml_string(xml)
    d2 = mujoco.MjData(m2)
    d2.qpos[:] = d.qpos
    mujoco.mj_kinematics(m2, d2)
    mujoco.mj_camlight(m2, d2)
    r = mujoco.Renderer(m2, CAM_H, CAM_W)
    r.enable_segmentation_rendering()
    r.update_scene(d2, camera="wrist")
    seg = r.render()
    r.close()
    vs, us = np.nonzero((seg[..., 0] == m2.geom("marker").id) & (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM))
    u, v = WristCam(m).project(d, P[None])[0]
    assert len(us) > 20 and np.hypot(us.mean() - u, vs.mean() - v) < 3
