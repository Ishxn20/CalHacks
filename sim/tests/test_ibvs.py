import numpy as np

from sim.ibvs import IBVS
from sim.servo_env import MAX_DQ, VisualReachEnv


def test_back_projection_recovers_the_grasp_point_with_the_nominal_mount():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=False, pool_size=4)
    ctl = IBVS()
    for s in range(4):
        obs, info = env.reset(seed=10 + s)
        est = ctl.estimate(obs)
        assert est is not None
        assert np.linalg.norm(est["xy"] - np.array(info["goal"][:2])) < 2e-3
        d = (est["yaw"] - info["goal"][3] + np.pi / 2) % np.pi - np.pi / 2       # jaw yaw, mod pi
        assert abs(np.degrees(d)) < 1.0


def test_ibvs_alone_reaches_noise_free_targets():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=False, pool_size=10)
    ctl, ok = IBVS(), 0
    for s in range(10):
        obs, _ = env.reset(seed=20 + s)
        ctl.reset()
        for _ in range(env.max_steps):
            obs, _, _, _, info = env.step(ctl(obs))
        ok += info["success"]
    assert ok >= 9


def test_mount_is_recovered_from_the_gripper_landmarks():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=1.0, pool_size=3)
    for s in range(3):
        obs, info = env.reset(seed=30 + s)
        ctl = IBVS()
        ctl.dq(obs)
        true = info["mount"]
        assert abs(ctl.mount["back"] - true["back"]) < 1e-3 and abs(ctl.mount["side"] - true["side"]) < 1e-3
        assert abs(ctl.mount["tilt_deg"] - true["tilt_deg"]) < 0.3


def test_ibvs_reaches_noise_free_targets_with_an_uncalibrated_mount():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=1.0, pool_size=10)
    ctl, ok = IBVS(), 0
    for s in range(10):
        obs, _ = env.reset(seed=40 + s)
        ctl.reset()
        for _ in range(env.max_steps):
            obs, _, _, _, info = env.step(ctl(obs))
        ok += info["success"]
    assert ok >= 9


def test_residual_env_with_zero_correction_is_ibvs():
    from sim.ibvs import ResidualEnv

    a = VisualReachEnv(seed=0, noise=False, mount_noise=1.0, pool_size=2)
    b = ResidualEnv(seed=0, noise=False, mount_noise=1.0, pool_size=2)
    obs_a, _ = a.reset(seed=50)
    obs_b, _ = b.reset(seed=50)
    assert obs_b.shape == b.observation_space.shape and np.allclose(obs_b[:len(obs_a)], obs_a)
    ctl = IBVS()
    for _ in range(20):
        obs_a, *_ = a.step(ctl(obs_a))
        obs_b, *_ = b.step(np.zeros(6))
    assert np.allclose(a.q, b.q, atol=1e-9)


def test_residual_correction_is_bounded():
    from sim.ibvs import RESIDUAL, ResidualEnv

    env = ResidualEnv(seed=0, noise=False, mount_noise=0.0, pool_size=1)
    env.reset(seed=51)
    base, q0 = env.base_dq.copy(), env.q.copy()
    env.step(np.ones(6))
    assert np.all(np.abs(env.q - q0 - base) <= RESIDUAL * MAX_DQ + 1e-9)
