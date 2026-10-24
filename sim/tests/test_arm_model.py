import mujoco
import numpy as np
import pytest

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics
from sim.arm_model import ARM_JOINTS, ASSETS, load_model, set_arm, tool_pose


@pytest.fixture(scope="module")
def model():
    return load_model()


def test_mujoco_fk_matches_pinocchio_within_1mm(model):
    kin, cfg, rng = Kinematics(), load_arm_config(), np.random.default_rng(0)
    data = mujoco.MjData(model)
    for _ in range(50):
        q = rng.uniform(cfg.lower, cfg.upper)
        set_arm(model, data, q)
        p, R = tool_pose(model, data)
        p_ref, R_ref = kin.fk(q)
        assert np.linalg.norm(p - p_ref) < 1e-3
        assert np.degrees(np.arccos(np.clip((np.trace(R.T @ R_ref) - 1) / 2, -1, 1))) < 0.5


def test_position_actuators_use_arm_yaml_gains_and_limits(model):
    cfg = load_arm_config()
    for j, (name, lo, hi) in enumerate(zip(ARM_JOINTS, cfg.lower, cfg.upper)):
        a = model.actuator(name)
        assert a.gainprm[0] == pytest.approx(cfg.joints[j].kp)
        assert a.ctrlrange == pytest.approx([lo, hi], abs=1e-5)
    assert model.actuator("gripper").trntype == mujoco.mjtTrn.mjTRN_JOINT


def test_vendored_meshes_are_small():
    assert sum(p.stat().st_size for p in ASSETS.rglob("*.stl")) < 3_000_000
