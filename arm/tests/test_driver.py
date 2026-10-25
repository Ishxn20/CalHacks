import math
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


def test_limit_check_tolerates_encoder_noise_at_the_stop(app):
    q = app.driver.q_command()
    q[1] = math.radians(-0.1)  # real arm at rest on its zero stop
    app.driver.check_limits(q)
    q[1] = math.radians(-1.5)
    with pytest.raises(ArmFault, match="joint2"):
        app.driver.check_limits(q)


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


def test_exact_move_cancels_gravity_sag(app):
    """The real arm sat 4 deg low on joint 2 and 2 deg short on joint 3 (tip 3 cm below its IK target)."""
    d = app.driver
    d.bus.sag[1:3] = np.radians([4.0, -2.0])
    q = d.q_command()
    q[1] += math.radians(10)
    d.move_to(q)                                                        # taught-pose replay: sags, as taught
    assert abs(math.degrees(d.q_measured()[1] - q[1]) - 4.0) < 0.5
    off = d.move_to(q, exact=True)
    assert np.degrees(np.abs(d.q_measured() - q)).max() < 0.3
    assert np.allclose(np.degrees(off[1:3]), [-4.0, 2.0], atol=0.3)


def test_exact_move_does_not_dip_by_the_sag_first(app):
    """After one sag correction, the next exact move must not first sag the full amount and climb back (on the real
    arm: a ~3 cm drop toward the table on arriving at the hover)."""
    import threading

    d = app.driver
    d.bus.sag[1] = math.radians(4.0)
    q = d.q_command()
    q[1] += math.radians(10)
    d.move_to(q, exact=True)                          # learns the sag
    q2 = q.copy()
    q2[1] += math.radians(5)
    seen, done = [], threading.Event()

    def sample():
        while not done.is_set():
            seen.append(d.q_measured()[1])
            done.wait(0.002)

    th = threading.Thread(target=sample)
    th.start()
    d.move_to(q2, exact=True)
    done.set()
    th.join()
    assert max(seen) < q2[1] + math.radians(1.0)      # never ran 4 deg past the target before correcting
