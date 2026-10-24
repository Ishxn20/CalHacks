import numpy as np
import pytest

from sim.servo_env import MAX_DQ, N_OBS, VisualReachEnv
from sim.wrist_cam import CAM_H, CAM_W


@pytest.fixture(scope="module")
def env():
    return VisualReachEnv(seed=0, noise=False, mount_noise=False, pool_size=4)


def test_reset_observes_camera_features_not_the_world_pose(env):
    obs, info = env.reset(seed=1)
    assert obs.shape == (N_OBS,) and np.all(np.isfinite(obs))
    gx, gy, gz, _ = info["goal"]
    for v in (gx, gy, gz):                       # the goal's world coordinates appear nowhere in the observation
        assert not np.any(np.isclose(obs, v, atol=1e-6))


def test_noise_free_features_are_the_projection_relative_to_the_fingertips(env):
    env.reset(seed=2)
    f = env.features()
    if not f["visible"]:
        pytest.skip("target out of view at this start")
    g, tip = env.cam.project(env.data, np.array([env.grasp_world(), env.data.site("tool").xpos]))
    assert f["offset"] == pytest.approx([(g[0] - tip[0]) / CAM_W, (g[1] - tip[1]) / CAM_H], abs=1e-6)


def test_target_in_the_gripper_region_is_occluded(env):
    env.reset(seed=3)
    tip = env.data.site("tool").xpos.copy()
    env.target["grasp"] = tip[:2]                 # right under the fingertips: hidden by the fingers
    env.target["z"] = tip[2]
    assert not env.features()["visible"]


def test_invisible_target_holds_the_last_observation_and_clears_the_flag():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=False, pool_size=2)
    obs0, _ = env.reset(seed=4)
    env.dropout = 1.0                              # every frame lost from now on
    obs1, *_ = env.step(np.zeros(6))
    vis = env.OBS_VISIBLE
    assert obs1[vis] == 0.0
    feats = slice(vis + 1, vis + 8)
    assert np.allclose(obs1[feats], obs0[feats])


def test_at_the_goal_pose_is_success_for_either_jaw_direction(env):
    for flip in (0.0, np.pi):
        _, info = env.reset(seed=5)
        x, y, z, yaw = info["goal"]
        q = env.kin.ik_topdown(x, y, z, yaw + flip, env.q, (env.lo, env.hi), allow_flip=False)
        assert q is not None
        env.q = q
        _, r, _, _, step_info = env.step(np.zeros(6))
        assert step_info["success"] and r > 0


def test_small_actions_make_fine_moves(env):
    env.reset(seed=6)
    q0 = env.q.copy()
    env.step(np.full(6, 0.1))
    assert np.abs(env.q - q0).max() <= MAX_DQ * 0.1 ** 3 + 1e-12


def test_nudge_moves_the_target_10_to_30_mm_and_turns_it_partway():
    env = VisualReachEnv(seed=0, noise=False, mount_noise=False, pool_size=2, nudge=True)
    _, info = env.reset(seed=7)
    g0 = np.array(info["goal"][:2])
    yaw0, moved = info["goal"][3], None
    for _ in range(env.max_steps):
        *_, step_info = env.step(np.zeros(6))
        if step_info.get("nudged"):
            moved, turned = np.linalg.norm(np.array(env.goal[:2]) - g0), abs(env.goal[3] - yaw0)
            break
    assert moved is not None and 0.010 <= moved <= 0.030 + 1e-9 and turned <= np.radians(15) + 1e-9
    assert 30 <= env.steps <= 80
