import math
import re

import numpy as np
import pytest

from scrub_arm.app import build
from scrub_arm.handoff import TaskFailed


@pytest.fixture
def ik_app(config_dir):
    p = config_dir / "arm.yaml"
    p.write_text(re.sub(r"^(\s*mode:\s*)slots", r"\g<1>ik", p.read_text(), flags=re.M))
    a = build(sim=True, config_dir=config_dir)
    m = a.cfg.motion
    m.speed_scale, m.approach_speed_scale, m.min_move_s, m.settle_timeout_s = 20.0, 1.0, 0.02, 0.1
    a.cfg.gripper.settle_s = 0.05
    a.cfg.control_rate_hz = 500
    a.driver.start()
    a.driver.enable()
    yield a
    a.driver.shutdown()


def test_ik_mode_builds_kinematics(ik_app):
    assert ik_app.assistant.kin is not None


def test_pick_at_sets_picked_from(ik_app):
    ik_app.assistant._pick_at(300, 0, 0, 4.0)
    assert ik_app.assistant._picked_from == (300, 0, 0, 4.0)


def test_pick_at_miss_raises(ik_app):
    ik_app.driver.bus.object_in_jaws = False
    with pytest.raises(TaskFailed):
        ik_app.assistant._pick_at(300, 0, 0, 4.0)


def test_unreachable_does_not_move(ik_app):
    q0 = ik_app.driver.q_command()
    with pytest.raises(TaskFailed):
        ik_app.assistant._pick_at(1500, 0, 0, 4.0)
    assert np.allclose(ik_app.driver.q_command(), q0)


def _contact_after(ik_app, n_moves):
    """Shove joint2 off its setpoint after the n-th move_to call (simulates fingertips landing on the pile)."""
    real = ik_app.driver.move_to
    calls = {"n": 0}

    def move_to(q, speed=1.0, wait=True):
        real(q, speed, wait)
        calls["n"] += 1
        if calls["n"] == n_moves:
            ik_app.driver.bus.external_offset[1] = math.radians(3.0)

    ik_app.driver.move_to = move_to


def test_descend_stops_1mm_above_contact(ik_app):
    a, pk = ik_app.assistant, ik_app.cfg.pick
    _contact_after(ik_app, 3)   # hover move + 2 steps: contact at the 2nd descent step
    z = a._descend_until_contact(300, 0, 0, pk.table_z)
    contact_z = pk.table_z + pk.hover - 2 * pk.descend_step
    assert z == pytest.approx(contact_z + 0.001)       # spec: back off 1 mm, not a whole step


def test_contact_on_first_step_is_an_obstacle_not_a_grasp(ik_app):
    a, pk = ik_app.assistant, ik_app.cfg.pick
    _contact_after(ik_app, 2)   # hover move, then contact on the very first descent step
    with pytest.raises(TaskFailed):
        a._descend_until_contact(300, 0, 0, pk.table_z)


def test_no_contact_reaches_floor(ik_app):
    a, pk = ik_app.assistant, ik_app.cfg.pick
    z = a._descend_until_contact(300, 0, 0, 0.010)
    assert z == pytest.approx(0.010, abs=1e-9)


def test_place_at_opens_gripper(ik_app):
    a = ik_app.assistant
    a._pick_at(300, 0, 0, 4.0)
    a._place_at(250, -200, 0, 4.0)
    assert abs(ik_app.driver.gripper_measured() - ik_app.cfg.gripper.open) < math.radians(2)


def test_place_with_contact_stops_on_the_pile(ik_app):
    a, pk = ik_app.assistant, ik_app.cfg.pick
    a._pick_at(300, 0, 0, 4.0)
    _contact_after(ik_app, 4)      # hover, then contact a couple of steps down: the pile is there
    a._place_at(300, 0, 0, 4.0, sense_contact=True)
    assert a.holding is None
    assert a.kin.fk(ik_app.driver.q_command())[0][2] > pk.table_z + 0.03   # back up at hover, never pushed down


def test_grasp_keeps_tip_direction(ik_app):
    a = ik_app.assistant
    for yaw in (10, 100, 190, 280):
        q = a._solve(300, 0, ik_app.cfg.pick.table_z + 0.05, yaw, allow_flip=False)
        _, R = a.kin.fk(q)
        jaw = R @ a.kin.jaw_axis_local
        assert np.dot(jaw[:2], [np.cos(np.radians(yaw)), np.sin(np.radians(yaw))]) > 0.999
