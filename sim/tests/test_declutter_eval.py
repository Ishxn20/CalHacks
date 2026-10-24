import numpy as np

from sim.declutter_eval import run_episode, summarise


def test_target_alone_is_grasped_first_try():
    r = run_episode(np.random.default_rng(0), items=["forceps_resano"], target="forceps_resano")
    assert r["outcome"] == "grasped" and r["moves"] == 0 and r["grasped_id"] == "forceps_resano"


def test_episode_reports_moves_and_outcome():
    r = run_episode(np.random.default_rng(3), n_items=6)
    assert r["outcome"] in {"grasped", "fail", "cap", "wrong"}
    assert 0 <= r["moves"] <= 6


def test_summary_counts():
    s = summarise([{"outcome": "grasped", "moves": 1, "say": ""}, {"outcome": "fail", "moves": 0, "say": "x"}])
    assert s["n"] == 2 and s["success_rate"] == 0.5 and s["mean_moves_success"] == 1.0
