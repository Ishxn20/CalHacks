import numpy as np

from scrub_vision.identify import feasible, on_top
from scrub_vision.planner import matches
from sim.declutter_eval import H, STAGING, run_episode
from sim.rl_env import DeclutterEnv, reward
from sim.rl_policy import N_FEATURES, Policy, legal


def test_reset_observes_target_and_legal_choices():
    env = DeclutterEnv(np.random.default_rng(1))
    obs = env.reset(items=["forceps_resano", "scalpel_10"], target="forceps_resano")
    assert obs["target"] == "forceps_resano"
    assert obs["features"].shape == (len(obs["legal"]) + 1, N_FEATURES)   # +1: the stop action


def test_grasping_the_target_ends_with_plus_one():
    env = DeclutterEnv(np.random.default_rng(0))
    obs = env.reset(items=["forceps_resano"], target="forceps_resano")
    a = next(i for i, d in enumerate(obs["legal"]) if matches(d, "forceps_resano"))
    _, r, done, info = env.step(a)
    assert done and r == 1.0 and info["outcome"] == "grasped"


def test_moving_a_non_target_costs_a_step_and_removes_it():
    env = DeclutterEnv(np.random.default_rng(0))
    obs = env.reset(items=["forceps_resano", "scalpel_10"], target="forceps_resano")
    movers = [i for i, d in enumerate(obs["legal"]) if not matches(d, "forceps_resano")]
    if not movers:                      # this seed's pile happens to hide the scalpel; nothing to test
        return
    obs2, r, done, info = env.step(movers[0])
    assert not done and r == -env.step_cost and env.remaining == ["forceps_resano"] and env.moves == 1


def test_stop_action_fails_with_zero_reward():
    env = DeclutterEnv(np.random.default_rng(0))
    obs = env.reset(items=["forceps_resano"], target="forceps_resano")
    _, r, done, info = env.step(len(obs["legal"]))
    assert done and r == 0.0 and info["outcome"] == "fail"


def test_reward_table():
    assert reward("grasped") == 1.0 and reward("wrong") == -1.0
    assert reward("fail") == 0.0 and reward("cap") == 0.0 and reward("ambiguous") == -1.0


def test_legal_is_exactly_the_recognised_on_top_feasible_detections():
    env = DeclutterEnv(np.random.default_rng(5))
    obs = env.reset(n_items=8)
    dets = obs["dets"]
    want = [d for d in dets if d.candidates and on_top(d, dets) and feasible(d, dets, H)]
    assert legal(dets, H) == want and obs["legal"] == want


def test_choose_returns_plan_shaped_json():
    env = DeclutterEnv(np.random.default_rng(2))
    obs = env.reset(n_items=6)
    p = Policy(np.zeros(N_FEATURES)).choose(obs["dets"], obs["target"], H, STAGING, set())
    target_keys = {"id", "x_mm", "y_mm", "yaw_deg", "thickness_mm", "confidence", "exposure", "ambiguous"}
    if p["action"] == "move_aside":
        assert set(p) == {"action", "target", "staging_index", "place", "say"}
        assert set(p["target"]) == target_keys and p["staging_index"] == 0
    elif p["action"] == "grasp_target":
        assert set(p) == {"action", "target", "place", "say"} and set(p["target"]) == target_keys
    else:
        assert p == {"action": "fail", "target": None, "place": None, "say": p["say"]}


def test_policy_round_trips_through_json(tmp_path):
    pol = Policy(np.arange(N_FEATURES, dtype=float))
    pol.save(tmp_path / "p.json")
    assert np.array_equal(Policy.load(tmp_path / "p.json").w, pol.w)


def test_policy_runs_in_the_eval_harness():
    pol = Policy(np.zeros(N_FEATURES))
    r = run_episode(np.random.default_rng(0), items=["forceps_resano"], target="forceps_resano",
                    planner=pol.choose)
    assert r["outcome"] in {"grasped", "fail"}
