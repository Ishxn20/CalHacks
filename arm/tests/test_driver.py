import math
import threading
import time

import numpy as np
import pytest

from scrub_arm.bus import SimBus
from scrub_arm.config import load_arm_config, load_poses
from scrub_arm.driver import ArmDriver, ArmFault, MinJerk, min_jerk_duration


def test_min_jerk_endpoints_and_peak_velocity():
    q0, q1 = np.zeros(6), np.radians([90, -30, 10, 0, 0, 45])
    vmax = np.radians([90] * 6)
    T = min_jerk_duration(q1 - q0, vmax, 0.1)
    tr = MinJerk(q0, q1, 0.0, T)
    assert np.allclose(tr.sample(0.0)[0], q0)
    q_end, qd_end, done = tr.sample(T)
    assert done and np.allclose(q_end, q1) and np.allclose(qd_end, 0)
    peak = max(np.max(np.abs(tr.sample(t)[1]) / vmax) for t in np.linspace(0, T, 400))
    assert peak <= 1.0 + 1e-6


def test_rejects_targets_outside_joint_limits(app):
    q = app.driver.q_command()
    q[1] = math.radians(-20)  # joint2 lower limit is 0
    with pytest.raises(ArmFault, match="joint2"):
        app.driver.move_to(q)


def test_tracking_error_freezes_arm(app):
    d = app.driver
    time.sleep(0.55)  # past the post-enable grace window
    d.bus.external_offset[2] = math.radians(40)  # something shoved joint3
    time.sleep(0.05)
    assert d.fault and "joint3" in d.fault
    with pytest.raises(ArmFault):
        d.move_pose("handoff")
    d.bus.external_offset[:] = 0
    d.clear_fault()
    d.move_pose("handoff")


def test_stale_feedback_right_after_enable_does_not_fault(app):
    d = app.driver
    d.limp()
    d.enable()
    # Cached state still reports where the arm was before it was dragged.
    d.bus.external_offset[1] = math.radians(40)
    time.sleep(0.1)
    d.bus.external_offset[:] = 0
    time.sleep(0.5)
    assert d.fault is None


def test_gripper_torque_stays_capped_on_long_travel(app):
    d, g = app.driver, app.cfg.gripper
    g.open, g.closed = math.radians(260), 0.0  # the real B601-RS jaw range
    d.bus.object_in_jaws = False
    worst = {"move": 0.0, "grip": 0.0}
    stop = False

    def watch():
        while not stop:
            with d._lock:
                _, kp, _, _, tau_ff = d._g_cmd
                torque = kp * abs(d._g_sp - d._pos[-1]) + abs(tau_ff)
            kind = "grip" if kp == 0.0 else "move"
            worst[kind] = max(worst[kind], torque)
            time.sleep(0.001)

    t = threading.Thread(target=watch)
    t.start()
    try:
        d.gripper_open()
        assert abs(d.gripper_measured() - g.open) < math.radians(2)
        d.bus.object_in_jaws = True  # jaws will stall on the instrument
        assert d.gripper_grasp()
    finally:
        stop = True
        t.join()
    assert worst["move"] <= g.move_torque + 0.05
    assert 0 < worst["grip"] <= g.grip_torque + 1e-9


def test_grasp_on_nothing_reports_empty_and_stops_pushing(app):
    d, g = app.driver, app.cfg.gripper
    d.bus.object_in_jaws = False
    d.gripper_open()
    assert not d.gripper_grasp()
    assert d._g_cmd[1] > 0 and d._g_cmd[4] == 0.0  # back to position hold, no squeeze into the stop


def test_real_arm_refuses_placeholder_poses_and_uncalibrated_gripper(config_dir):
    cfg = load_arm_config(config_dir / "arm.yaml")
    poses = load_poses(config_dir / "poses.yaml")
    d = ArmDriver(cfg, SimBus(cfg, start=np.append(poses.get("home").q, 0.0)), poses, sim=False)
    d.start()
    d.enable()
    try:
        with pytest.raises(ArmFault, match="placeholder"):
            d.move_pose("handoff")
        with pytest.raises(ArmFault, match="gripper"):
            d.gripper_grasp()
    finally:
        d.shutdown()


def test_enable_holds_current_position(config_dir):
    cfg = load_arm_config(config_dir / "arm.yaml")
    start = np.radians([10, 40, 50, 5, -5, 20, 0])
    bus = SimBus(cfg, start=start)
    d = ArmDriver(cfg, bus, load_poses(config_dir / "poses.yaml"), sim=True)
    d.start()
    d.enable()
    time.sleep(0.1)
    try:
        assert np.allclose(bus.pos, start, atol=1e-6)
    finally:
        d.shutdown()


def test_samples_are_taken_at_about_20_hz_and_drained_by_time(app):
    d = app.driver
    t0 = time.time()
    time.sleep(1.0)
    got = d.samples(t0)
    assert 10 <= len(got) <= 40                  # ~20 Hz, generous for a loaded CI box
    s = got[-1]
    assert len(s["q"]) == len(s["q_cmd"]) == len(s["tau"]) == 6 and "gripper" in s and "moving" in s
    assert all(x is None or isinstance(x, float) for x in s["tau"])
    assert d.samples(got[-1]["t"]) == [] or d.samples(got[-1]["t"])[0]["t"] > got[-1]["t"]
    assert "tau_nm" in d.snapshot() and "gripper_tau_nm" in d.snapshot()
