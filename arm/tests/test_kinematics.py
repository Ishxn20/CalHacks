import numpy as np
import pytest

from scrub_arm.config import load_arm_config
from scrub_arm.kinematics import Kinematics


@pytest.fixture(scope="module")
def kin():
    return Kinematics()


@pytest.fixture(scope="module")
def limits():
    cfg = load_arm_config()
    return cfg.lower, cfg.upper


SEED = np.radians([0, 60, 70, 0, 50, 0])


def test_fk_zero_pose_is_in_front_of_base(kin):
    p, R = kin.fk(np.zeros(6))
    assert p[0] > 0.25 and abs(p[1]) < 0.01 and p[2] > 0.15


def test_ik_tool_points_down(kin, limits):
    q = kin.ik_topdown(0.30, 0.0, 0.05, 0.0, SEED, limits)
    assert q is not None
    _, R = kin.fk(q)
    assert np.dot(R @ kin.tool_axis_local, [0, 0, -1]) > 0.999


def test_ik_round_trip(kin, limits):
    q1 = kin.ik_topdown(0.28, 0.08, 0.06, np.radians(30), SEED, limits)
    assert q1 is not None
    p, _ = kin.fk(q1)
    q2 = kin.ik_topdown(*p, np.radians(30), q1 + 0.05, limits)
    assert q2 is not None
    assert np.linalg.norm(kin.fk(q2)[0] - p) < 2e-3


def test_ik_yaw_sets_jaw_axis(kin, limits):
    yaw = np.radians(40)
    q = kin.ik_topdown(0.30, 0.0, 0.05, yaw, SEED, limits)
    _, R = kin.fk(q)
    jaw_world = R @ kin.jaw_axis_local
    # jaws separate along yaw or yaw+pi (pinch is symmetric)
    assert abs(abs(np.dot(jaw_world, [np.cos(yaw), np.sin(yaw), 0])) - 1) < 1e-3


def test_ik_unreachable_returns_none(kin, limits):
    assert kin.ik_topdown(1.5, 0.0, 0.05, 0.0, np.zeros(6), limits) is None


def test_ik_respects_limits(kin, limits):
    lo, hi = limits
    q = kin.ik_topdown(0.30, 0.0, 0.05, 0.0, SEED, limits)
    assert np.all(q >= lo - 1e-9) and np.all(q <= hi + 1e-9)
