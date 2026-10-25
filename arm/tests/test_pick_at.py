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

    def move_to(q, speed=1.0, wait=True, exact=False):
        off = real(q, speed, wait, exact)
        calls["n"] += 1
        if calls["n"] == n_moves:
            ik_app.driver.bus.external_offset[1] = math.radians(3.0)
        return off

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


def test_pick_at_descends_where_refine_aligned(ik_app):
    """pick.reach rl: after the IK hover, a refine hook (the camera-guided policy) may correct the target;
    the contact descent then goes straight down at the corrected point."""
    calls = []
    ik_app.assistant.refine = lambda x, y, yaw, th: calls.append((x, y, yaw, th)) or (310.0, 5.0, 10.0)
    ik_app.assistant._pick_at(300, 0, 0, 4.0)
    assert calls == [(300, 0, 0, 4.0)]
    assert ik_app.assistant._picked_from == (310.0, 5.0, 10.0, 4.0)
    p, _ = ik_app.assistant.kin.fk(ik_app.driver.q_command())
    assert abs(p[0] * 1000 - 310) < 2 and abs(p[1] * 1000 - 5) < 2


def test_descent_lands_where_ik_aimed_despite_sag(ik_app):
    a, d = ik_app.assistant, ik_app.driver
    d.bus.sag[1:3] = np.radians([4.0, -2.0])
    z_floor = ik_app.cfg.pick.table_z + 0.004
    d.move_to(a._solve(300, 0, ik_app.cfg.pick.table_z + ik_app.cfg.pick.hover, 0))
    a._descend_until_contact(300, 0, 0, z_floor)
    p, _ = a.kin.fk(d.q_measured())
    assert abs(p[0] * 1000 - 300) < 2 and abs(p[1] * 1000) < 2 and abs(p[2] - z_floor) < 0.002


def test_place_puts_down_at_the_planned_height_despite_sag(ik_app):
    a, d = ik_app.assistant, ik_app.driver
    d.bus.sag[1:3] = np.radians([4.0, -2.0])
    seen = []
    orig = d.gripper_open
    d.gripper_open = lambda *aa, **kw: seen.append(a.kin.fk(d.q_measured())[0].copy()) or orig(*aa, **kw)
    a._place_at(300, 0, 0, 4.0)
    z_place = ik_app.cfg.pick.table_z + 0.002 + 0.005
    assert seen and abs(seen[0][2] - z_place) < 0.002 and abs(seen[0][0] * 1000 - 300) < 2


def test_move_line_goes_straight_in_x_only(ik_app):
    """'Move X by 10': y, z and the gripper's orientation stay put the whole way (joint-space moves between IK
    poses dipped and swung the gripper on the real arm), and it's one smooth motion, not a stop every 2 mm."""
    import threading

    a, d = ik_app.assistant, ik_app.driver
    d.bus.sag[1:3] = np.radians([4.0, -2.0])
    d.move_to(a._solve(320, 10, ik_app.cfg.pick.table_z + 0.10, 30), exact=True)
    p0, R0 = a.kin.fk(d.q_measured())
    seen, done = [], threading.Event()

    def sample():                                       # where the gripper actually is during the move
        while not done.is_set():
            seen.append(a.kin.fk(d.q_measured())[0].copy())
            done.wait(0.002)

    calls = {"n": 0}
    real = d.move_to

    def count(*args, **kw):
        calls["n"] += 1
        return real(*args, **kw)

    d.move_to = count
    th = threading.Thread(target=sample)
    th.start()
    a.move_line(30, 0, 0)
    done.set()
    th.join()
    d.move_to = real
    pts = np.array(seen)
    assert len(pts) > 20 and calls["n"] <= 2                   # one smooth path (+ the exact landing)
    assert np.abs(pts[:, 1] - p0[1]).max() < 1e-3 and np.abs(pts[:, 2] - p0[2]).max() < 1e-3
    p1, R1 = a.kin.fk(d.q_measured())
    assert abs(p1[0] - p0[0] - 0.030) < 5e-4
    assert np.degrees(np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2, -1, 1))) < 1.0


def test_move_line_turns_in_place(ik_app):
    """Turning the jaws at the hover must not move the fingertips (rotation about the vertical through them)."""
    a, d = ik_app.assistant, ik_app.driver
    d.move_to(a._solve(320, 10, ik_app.cfg.pick.table_z + 0.10, 30), exact=True)
    p0, R0 = a.kin.fk(d.q_measured())
    a.move_line(0, 0, 0, dyaw_deg=20)
    p1, R1 = a.kin.fk(d.q_measured())
    assert np.linalg.norm(p1 - p0) < 1e-3
    j0, j1 = R0 @ a.kin.jaw_axis_local, R1 @ a.kin.jaw_axis_local
    turn = math.degrees(math.atan2(j1[1], j1[0]) - math.atan2(j0[1], j0[0]))
    tilt0, tilt1 = (R0 @ a.kin.tool_axis_local)[2], (R1 @ a.kin.tool_axis_local)[2]
    assert abs(turn - 20) < 1.0 and abs(tilt1 - tilt0) < 1e-3        # same tilt (straight down or the pick tilt)


def test_descent_starts_where_the_staged_centring_finished(ik_app):
    """refine may return the height it ended at (the 3 cm stage): the contact descent continues from there instead
    of going back up to the hover; the lift afterwards is straight up."""
    a, d, pk = ik_app.assistant, ik_app.driver, ik_app.cfg.pick
    z3 = pk.table_z + 0.03

    def refine(x, y, yaw, th):
        d.move_to(a._solve(310, 5, z3, 10), exact=True)                    # centring left it here
        return 310.0, 5.0, 10.0, z3

    a.refine = refine
    zs = []
    real = d.move_to

    def spy(q, speed=1.0, wait=True, exact=False):
        off = real(q, speed, wait, exact)
        zs.append(a.kin.fk(d.q_measured())[0].copy())
        return off

    d.move_to = spy
    a._pick_at(300, 0, 0, 4.0)
    d.move_to = real
    pts = np.array(zs)
    after = pts[int(np.argmin(np.abs(pts[:, 2] - z3))):]                  # from the 3 cm stage on
    down = after[: int(np.argmin(after[:, 2])) + 1]
    assert down[:, 2].max() < z3 + 0.002                                   # never climbed back to the hover
    up = after[int(np.argmin(after[:, 2])):]
    assert np.abs(up[:, :2] - up[0, :2]).max() < 1e-3                    # lifted straight up


def test_tilted_pick_keeps_its_tilt_to_the_table(ik_app):
    """pick.tilt_pose: hover and grab share the scan pose's tilt (the wrist camera looks straight down, so a vertical
    descent barely moves the target in the image); where the tilt cannot reach both ends, straight down."""
    a, k = ik_app.assistant, ik_app.assistant.kin
    tz = (k.tilt_R @ k.tool_axis_local)[2]
    for z in (0.10, 0.03, 0.003):
        _, R = k.fk(a._solve(440, 0, ik_app.cfg.pick.table_z + z, 90))
        assert abs((R @ k.tool_axis_local)[2] - tz) < 1e-3
    assert a._tilt_for(440, 0, 90, 0.003, 0.10) and not a._tilt_for(200, 0, 90, 0.003, 0.10)
    _, R = k.fk(a._solve(200, 0, ik_app.cfg.pick.table_z + 0.10, 90, tilt=False))   # where the tilt cannot reach
    assert (R @ k.tool_axis_local)[2] < -0.999
