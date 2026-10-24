import numpy as np
import pytest

from scrub_arm.config import load_arm_config
from sim.reach_env import MAX_DQ, ReachEnv


@pytest.fixture(scope="module")
def env():
    return ReachEnv(seed=0)


def test_reset_gives_a_reachable_topdown_target(env):
    obs, info = env.reset(seed=1)
    assert obs.shape == env.observation_space.shape and np.all(np.isfinite(obs))
    x, y, z, yaw = info["target"]
    assert 0.12 <= np.hypot(x, y) <= 0.40 and env.table_z + 0.005 <= z <= env.table_z + 0.125
    assert info["q_ik"] is not None          # every target is one ik_topdown can reach


def test_action_is_a_small_clipped_joint_delta(env):
    env.reset(seed=2)
    q0 = env.q.copy()
    env.step(np.full(6, 5.0))                # out of range: clipped to +MAX_DQ per joint
    cfg = load_arm_config()
    assert np.all(env.q - q0 <= MAX_DQ + 1e-9)
    assert np.all(env.q >= cfg.lower) and np.all(env.q <= cfg.upper)


def test_arriving_at_the_target_gives_the_bonus_and_ends(env):
    _, info = env.reset(seed=3)
    q_goal = info["q_ik"]
    for _ in range(200):
        _, r, done, _, step_info = env.step(np.clip((q_goal - env.q) / MAX_DQ, -1, 1))
        if done:
            break
    assert done and step_info["success"] and r > 0
    assert step_info["pos_err"] < 3e-3 and step_info["rot_err_deg"] < 3


def test_far_from_target_is_negative_reward(env):
    env.reset(seed=4)
    _, r, done, _, info = env.step(np.zeros(6))
    assert not info["success"] and r < 0


def test_touching_the_table_is_penalised(env):
    env.reset(seed=5)
    env.q = np.radians([0, 120, 20, 0, 60, 0])     # wrist driven into the table
    _, r_hit, _, _, info = env.step(np.zeros(6))
    assert info["table"] and r_hit < -env.table_penalty + 0.0
