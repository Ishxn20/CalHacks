# Saved grasp-selection models (snapshot 2026-10-24 01:4x)

| File | What | Held-out result |
|---|---|---|
| `rl_grasp_policy_run1.json` | RL policy (linear softmax over per-detection features, REINFORCE), run 1 best checkpoint, iter 25 | 94.2% right, 4.2% wrong (120 val piles) |
| `rl_grasp_policy_run2_best.json` | RL policy, run 2 best checkpoint so far (run still training) | see file `val_success` / `val_wrong` |
| `rl_training_curve_run2_partial.csv` | run 2 curve up to the snapshot | |
| `rules_planner_eval_200.json` | rule-based planner (`perception/scrub_vision/planner.py`, git tag `rules-planner-v1`) on 200 MuJoCo piles, seed 0 | 94% right, 1% wrong; ~97% counting any scalpel as right (decision made after this run) |

The RL weights load with `sim/rl_policy.py` on branch `claude/rl-grasp-policy-f51e28` (`Policy.load(path)`).
Both evaluations used exact instance masks, not the trained segmentation model.
