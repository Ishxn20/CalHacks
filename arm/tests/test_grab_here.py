import math
import re
import threading

import numpy as np
import pytest

from scrub_arm.app import build
from scrub_arm.handoff import TaskFailed

TILTED_LOW = np.radians([0, 68, 46, -48, 0, 0])     # tip ~74 mm up, fingers 20 deg off vertical
TOO_HIGH = np.radians([0, 60, 50, -20, 0, 0])       # tip ~240 mm up: can't point down there


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


def _tip(app):
    return app.assistant.kin.fk(app.driver.q_command())[0]


def test_pivot_keeps_the_tip_still_the_whole_way(ik_app):
    a, d, kin = ik_app.assistant, ik_app.driver, ik_app.assistant.kin
    d.move_to(TILTED_LOW)
    start = _tip(ik_app)
    worst = {"mm": 0.0}
    done = threading.Event()

    def watch():  # sample the commanded tip position while the arm is moving
        while not done.is_set():
            worst["mm"] = max(worst["mm"], float(np.linalg.norm(_tip(ik_app) - start)) * 1000)

    t = threading.Thread(target=watch)
    t.start()
    try:
        x, y, z, yaw = a.pivot_down()
    finally:
        done.set()
        t.join()
    assert math.degrees(kin.tilt_from_down(d.q_command())) < 0.5
    assert worst["mm"] < 1.0
    assert (x, y, z) == pytest.approx((start[0] * 1000, start[1] * 1000, start[2]), abs=1e-6)


def test_pivot_turns_the_jaws_to_the_requested_direction(ik_app):
    a, d, kin = ik_app.assistant, ik_app.driver, ik_app.assistant.kin
    d.move_to(TILTED_LOW)
    _, _, _, yaw = a.pivot_down(yaw_deg=30.0)
    _, R = kin.fk(d.q_command())
    jaw = R @ kin.jaw_axis_local
    # a pinch is symmetric, so 30 or 210 deg are both right
    assert abs(math.sin(math.atan2(jaw[1], jaw[0]) - math.radians(30))) < 0.02
    assert abs(math.sin(math.radians(yaw - 30))) < 1e-6


def test_pivot_refuses_up_high_without_moving(ik_app):
    a, d = ik_app.assistant, ik_app.driver
    d.move_to(TOO_HIGH)
    q0 = d.q_command()
    with pytest.raises(TaskFailed, match="tip"):
        a.pivot_down()
    assert np.allclose(d.q_command(), q0)


def test_grab_here_picks_up_whatever_is_under_the_tip(ik_app):
    a, d, kin = ik_app.assistant, ik_app.driver, ik_app.assistant.kin
    d.move_to(TILTED_LOW)
    start = _tip(ik_app)
    a.grab_here(thickness_mm=4.0)
    assert d.gripper_holding()
    x, y, yaw, thick = a._picked_from
    assert (x, y) == pytest.approx((start[0] * 1000, start[1] * 1000), abs=0.5)
    # back up at the starting height, pointing down, same spot
    assert np.allclose(_tip(ik_app), start, atol=1e-3)
    assert math.degrees(kin.tilt_from_down(d.q_command())) < 0.5


def test_grab_here_miss_raises_and_opens(ik_app):
    ik_app.driver.bus.object_in_jaws = False
    ik_app.driver.move_to(TILTED_LOW)
    with pytest.raises(TaskFailed, match="missed"):
        ik_app.assistant.grab_here(thickness_mm=4.0)
