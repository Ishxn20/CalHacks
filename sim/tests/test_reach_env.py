import numpy as np
import pytest

from scrub_arm.config import load_arm_config
from sim.reach_env import MAX_DQ, ReachEnv, step_scale


@pytest.fixture(scope="module")
def env():
    return ReachEnv(seed=0)


def test_reset_gives_a_reachable_topdown_target_as_a_goal_dict(env):
    obs, info = env.reset(seed=1)
    assert set(obs) == {"observation", "achieved_goal", "desired_goal"}
    for k, space in env.observation_space.spaces.items():
        assert obs[k].shape == space.shape and np.all(np.isfinite(obs[k]))
    x, y, z, yaw = info["target"]
    assert 0.12 <= np.hypot(x, y) <= 0.40 and env.table_z + 0.005 <= z <= env.table_z + 0.125
    assert info["q_ik"] is not None          # every target is one ik_topdown can reach
    assert env.max_steps == 150


def test_step_scale_shrinks_with_error():
    assert step_scale(0.2, 40.0) == 1.0                      # far: full 0.04 rad steps
    assert step_scale(0.01, 2.0) == pytest.approx(0.2)
    assert step_scale(0.0005, 0.1) == 0.03                   # close: floor, ~0.5 mm at the tool
    assert step_scale(0.001, 10.0) == pytest.approx(0.5)     # rotation error alone keeps it coarse


def test_effective_step_shrinks_as_the_error_shrinks(env):
    _, info = env.reset(seed=2)
    q_goal = info["q_ik"]
    far = env.q.copy()
    env.step(np.ones(6))
    big = np.abs(env.q - far).max()
    env.set_target(info["target"], q_goal + 0.001)            # ~0.5 mm away
    near = env.q.copy()
    obs, *_ = env.step(np.ones(6))
    small = np.abs(env.q - near).max()
    assert small < 0.1 * big and small <= MAX_DQ * 0.05
    assert obs["observation"][6] == pytest.approx(env.scale)  # the scale is observed


def test_action_is_clipped_and_stays_in_limits(env):
    env.reset(seed=3)
    q0 = env.q.copy()
    env.step(np.full(6, 5.0))
    cfg = load_arm_config()
    assert np.all(env.q - q0 <= MAX_DQ + 1e-9)
    assert np.all(env.q >= cfg.lower) and np.all(env.q <= cfg.upper)


def test_arriving_gives_the_bonus_and_success(env):
    _, info = env.reset(seed=4)
    q_goal = info["q_ik"]
    for _ in range(300):
        _, r, terminated, truncated, step_info = env.step(np.clip((q_goal - env.q) / (MAX_DQ * env.scale), -1, 1))
        if step_info["success"]:
            break
    assert step_info["success"] and not terminated and r > 0
    assert step_info["pos_err"] < 3e-3 and step_info["rot_err_deg"] < 3


def test_reward_is_steeper_near_the_target(env):
    g = env.reset(seed=5)[0]["desired_goal"]
    def r(dx_mm):
        a = g.copy()
        a[0] += dx_mm / 100.0                 # goal positions are in decimetres
        return float(env.compute_reward(a, g, {"penalty": 0.0}))
    assert (r(1) - r(4)) / 3 > (r(4) - r(10)) / 6 > (r(10) - r(30)) / 20 > 0   # per mm, steeper when close


def test_compute_reward_matches_step_reward_and_vectorises(env):
    env.reset(seed=6)
    obs, r, *_, info = env.step(np.zeros(6))
    assert env.compute_reward(obs["achieved_goal"], obs["desired_goal"], info) == pytest.approx(r)
    batch = env.compute_reward(np.stack([obs["achieved_goal"]] * 3), np.stack([obs["desired_goal"]] * 3),
                               np.array([info] * 3))
    assert batch.shape == (3,) and np.allclose(batch, r)


def test_touching_the_table_is_penalised(env):
    env.reset(seed=7)
    env.q = np.radians([0, 120, 20, 0, 60, 0])     # wrist driven into the table
    _, r_hit, _, _, info = env.step(np.zeros(6))
    assert info["table"] and info["penalty"] >= env.table_penalty
