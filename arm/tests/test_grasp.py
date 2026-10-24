import math
import threading
import time

import numpy as np
import pytest

from scrub_arm.grasp import GrabError, grab_below, point_down, tilt_from_down
from scrub_arm.kinematics import IKError, Kinematics, rotvec

SEED = np.radians([0, 60, 60, 0, 60, 0])
# Gripper ~7 cm up, ~27 cm out, tilted 20 deg off vertical (q2 - q3 - q4 = 70, not 90).
ROUGHLY_DOWN = np.radians([0, 68, 46, -48, 0, 0])


@pytest.fixture
def kin(app):
    return app.kin


@pytest.fixture
def above_table(app, kin):
    """Gripper a few cm over the table, then pointed straight down."""
    app.driver.move_to(ROUGHLY_DOWN)
    point_down(app.driver, kin)
    return app


def tool_z(app, kin):
    return float(kin.fk(app.driver.q_command())[2, 3])


def test_jacobian_matches_finite_differences(kin):
    q = np.radians([10, 50, 60, 5, 30, -20])
    J, Jn, eps = kin.jacobian(q), np.zeros((6, 6)), 1e-6
    T0 = kin.fk(q)
    for i in range(6):
        dq = np.zeros(6)
        dq[i] = eps
        T1 = kin.fk(q + dq)
        Jn[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
        Jn[3:, i] = rotvec(T1[:3, :3] @ T0[:3, :3].T) / eps
    assert np.allclose(J, Jn, atol=1e-5)


def test_ik_round_trip(kin):
    q_true = np.radians([20, 70, 80, -40, 30, 10])
    q = kin.ik(kin.fk(q_true), np.radians([0, 60, 60, 0, 60, 0]))
    assert np.allclose(kin.fk(q), kin.fk(q_true), atol=1e-3)


def test_vertical_line_keeps_xy_and_orientation(kin):
    T0 = kin.fk(SEED)
    qs = kin.line(SEED, np.array([0, 0, -0.05]))
    for q in qs:
        T = kin.fk(q)
        assert np.allclose(T[:2, 3], T0[:2, 3], atol=2e-4)
        assert np.linalg.norm(rotvec(T[:3, :3] @ T0[:3, :3].T)) < 2e-3
    assert kin.fk(qs[-1])[2, 3] == pytest.approx(T0[2, 3] - 0.05, abs=2e-4)


def test_unreachable_line_raises(kin):
    with pytest.raises(IKError):
        kin.line(SEED, np.array([0, 0, 2.0]))


def test_point_down(above_table, kin):
    assert math.degrees(tilt_from_down(kin, above_table.driver.q_command())) < 0.5


def test_grab_descends_grasps_and_lifts(above_table, kin):
    app = above_table
    z0 = tool_z(app, kin)
    res = grab_below(app.driver, kin, depth=0.05)
    assert res.holding and not res.contact
    assert res.grasp_z == pytest.approx(z0 - 0.05, abs=1e-3)
    assert tool_z(app, kin) == pytest.approx(res.grasp_z + app.cfg.grab.lift, abs=1e-3)


def test_grab_with_table_height_stops_at_clearance(above_table, kin):
    app = above_table
    table = tool_z(app, kin) - 0.06
    res = grab_below(app.driver, kin, table_z=table)
    assert res.grasp_z == pytest.approx(table + app.cfg.grab.clearance, abs=1e-3)


def test_grab_stops_on_contact_and_backs_off(above_table, kin):
    app = above_table
    d, bus = app.driver, app.driver.bus
    z0 = tool_z(app, kin)
    surface = z0 - 0.03
    done = threading.Event()

    def table():  # a floor at `surface`: the arm can't go lower, though its setpoint can
        while not done.is_set():
            measured_z = float(kin.fk(d.q_measured())[2, 3])
            # blocked while touching and still being pushed down; free again once the setpoint lifts
            bus.arm_blocked = measured_z <= surface and tool_z(app, kin) <= measured_z + 1e-4
            time.sleep(0.001)

    t = threading.Thread(target=table, daemon=True)
    t.start()
    try:
        res = grab_below(d, kin)  # no table height: contact is the only stop
    finally:
        done.set()
        t.join(timeout=5)
    assert res.contact and res.holding
    assert res.grasp_z == pytest.approx(surface + app.cfg.grab.backoff, abs=1.5e-3)
    assert d.fault is None


def test_grab_miss_reports_empty(above_table, kin):
    above_table.driver.bus.object_in_jaws = False
    res = grab_below(above_table.driver, kin, depth=0.03)
    assert not res.holding


def test_grab_refuses_when_not_pointing_down(app, kin):
    app.driver.move_to(np.radians([0, 60, 60, 0, 0, 0]))
    with pytest.raises(GrabError, match="off straight down"):
        grab_below(app.driver, kin, depth=0.03)


def test_descent_never_aims_below_the_floor(above_table, kin):
    app = above_table
    app.cfg.grab.floor_z = tool_z(app, kin) - 0.02   # pretend the table is 2 cm down
    res = grab_below(app.driver, kin, depth=0.10)     # asks for 10 cm
    assert res.grasp_z == pytest.approx(app.cfg.grab.floor_z, abs=1e-3)
