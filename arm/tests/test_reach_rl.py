import math
import re

import numpy as np
import pytest

from scrub_arm.app import build
from scrub_arm.config import load_arm_config
from scrub_arm.handoff import TaskFailed


def _rl_config(config_dir):
    p = config_dir / "arm.yaml"
    text = re.sub(r"^(\s*mode:\s*)slots", r"\g<1>ik", p.read_text(), flags=re.M)
    p.write_text(re.sub(r"^(\s*reach:\s*)ik", r"\g<1>rl", text, flags=re.M))
    return config_dir


class FakeReach:
    """Stands in for the trained policy: returns whatever joint vector the test wants."""

    def __init__(self, q):
        self.q, self.calls = q, 0

    def solve(self, x, y, z, yaw, q_seed, allow_flip=True):
        self.calls += 1
        return None if self.q is None else np.array(self.q, float)


@pytest.fixture
def rl_app(config_dir):
    a = build(sim=True, config_dir=_rl_config(config_dir))
    yield a
    a.driver.shutdown()


def _ik(app, x_mm, y_mm, z, yaw_deg):
    c = app.cfg
    return app.assistant.kin.ik_topdown(x_mm / 1000, y_mm / 1000, z, math.radians(yaw_deg),
                                        app.driver.q_command(), (c.lower, c.upper))


def test_reach_defaults_to_ik(config_dir):
    assert load_arm_config(config_dir / "arm.yaml").pick.reach == "ik"


def test_bad_reach_value_is_rejected(config_dir):
    p = config_dir / "arm.yaml"
    p.write_text(re.sub(r"^(\s*reach:\s*)ik", r"\g<1>magic", p.read_text(), flags=re.M))
    with pytest.raises(ValueError):
        load_arm_config(p)


def test_rl_mode_loads_the_reach_policy(rl_app):
    from sim.reach_policy import ReachPolicy

    assert isinstance(rl_app.assistant.reach, ReachPolicy) and rl_app.assistant.kin is not None


def test_rl_solution_passing_checks_is_used(rl_app):
    q = _ik(rl_app, 280, 40, 0.06, 30)
    rl_app.assistant.reach = FakeReach(q)
    assert np.allclose(rl_app.assistant._solve(280, 40, 0.06, 30), q)


def test_rl_never_falls_back_to_ik(rl_app, monkeypatch):
    rl_app.assistant.reach = FakeReach(None)
    monkeypatch.setattr(rl_app.assistant.kin, "ik_topdown", lambda *a, **k: pytest.fail("fell back to IK"))
    with pytest.raises(TaskFailed, match="reach policy"):
        rl_app.assistant._solve(280, 40, 0.06, 30)


def test_rl_solution_off_target_is_refused(rl_app):
    q = _ik(rl_app, 280, 40, 0.06, 30)
    rl_app.assistant.reach = FakeReach(q + np.radians([2, 0, 0, 0, 0, 0]))   # ~10 mm sideways
    with pytest.raises(TaskFailed, match="mm"):
        rl_app.assistant._solve(280, 40, 0.06, 30)


def test_rl_solution_wrong_yaw_is_refused(rl_app):
    q = _ik(rl_app, 280, 40, 0.06, 30)
    rl_app.assistant.reach = FakeReach(q + np.radians([0, 0, 0, 0, 0, 10]))
    with pytest.raises(TaskFailed, match="deg"):
        rl_app.assistant._solve(280, 40, 0.06, 30)


def test_rl_solution_outside_joint_limits_is_refused(rl_app):
    q = _ik(rl_app, 280, 40, 0.06, 30)
    q[1] = -0.5                                    # joint2 limit is 0..179 deg
    rl_app.assistant.reach = FakeReach(q)
    with pytest.raises(TaskFailed, match="limits"):
        rl_app.assistant._solve(280, 40, 0.06, 30)


def test_rl_refusal_does_not_move(rl_app):
    rl_app.driver.start()
    rl_app.driver.enable()
    q0 = rl_app.driver.q_command()
    rl_app.assistant.reach = FakeReach(None)
    with pytest.raises(TaskFailed):
        rl_app.assistant._pick_at(280, 40, 30, 4.0)
    assert np.allclose(rl_app.driver.q_command(), q0)
