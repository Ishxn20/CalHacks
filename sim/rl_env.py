"""The declutter loop of sim/declutter_eval.py as an RL environment (gymnasium-style reset/step).

Action = index into obs["legal"] (grasp it: grasp_target if it matches the request, else move it to staging),
or len(obs["legal"]) = stop. Reward: +1 target grasped, -1 wrong instrument grasped or target moved aside,
-step_cost per move, episode cap max_moves. Same rng consumption as run_episode(), so seeds line up.
"""
from __future__ import annotations

import numpy as np

from scrub_vision.identify import identify
from scrub_vision.planner import matches
from scrub_vision.render import render_scene, scene_mesh
from scrub_vision.templates import interchangeable

from sim.declutter_eval import CAM, H, _assets
from sim.pile_sim import settle_pile
from sim.rl_policy import features, legal

REWARD = {"grasped": 1.0, "wrong": -1.0, "ambiguous": -1.0, "fail": 0.0, "cap": 0.0}


def reward(outcome: str) -> float:
    return REWARD[outcome]


class DeclutterEnv:
    def __init__(self, rng: np.random.Generator, max_moves: int = 6, step_cost: float = 0.05) -> None:
        self.rng, self.max_moves, self.step_cost = rng, max_moves, step_cost

    def reset(self, n_items: int | None = None, items: list[str] | None = None, target: str | None = None) -> dict:
        insts, _ = _assets()
        if items is None:
            n = n_items or int(self.rng.integers(3, 9))
            items = [str(f) for f in self.rng.choice(list(insts), size=n, replace=False)]
        self.target = target or str(self.rng.choice(items))
        self.remaining, self.moves = list(items), 0
        return self._observe()

    def _observe(self) -> dict:
        insts, templates = _assets()
        pile, _ = settle_pile(self.remaining, self.rng)
        _, masks = render_scene([(scene_mesh(insts[f]), T) for f, T in pile], CAM, self.rng)
        self.dets, self.truth = [], {}
        for (fid, _), m in zip(pile, masks):
            if m.sum() < 300:
                continue
            d = identify(m, None, H, templates, insts)
            self.dets.append(d)
            self.truth[id(d)] = fid
        self.legal = legal(self.dets, H)
        return {"target": self.target, "dets": self.dets, "legal": self.legal,
                "features": features(self.legal, self.dets, self.target)}

    def step(self, action: int) -> tuple[dict | None, float, bool, dict]:
        if action == len(self.legal):
            return None, reward("fail"), True, {"outcome": "fail", "moves": self.moves}
        d = self.legal[action]
        actual = self.truth[id(d)]
        if matches(d, self.target):
            ok = actual in interchangeable(_assets()[0], self.target)
            outcome = "grasped" if ok else "wrong"
            return None, reward(outcome), True, {"outcome": outcome, "moves": self.moves, "grasped_id": actual}
        if actual == self.target:          # moved the target aside
            return None, reward("wrong"), True, {"outcome": "wrong", "moves": self.moves, "grasped_id": actual}
        self.remaining.remove(actual)
        self.moves += 1
        if self.moves >= self.max_moves:
            return None, reward("cap") - self.step_cost, True, {"outcome": "cap", "moves": self.moves}
        return self._observe(), -self.step_cost, False, {"moves": self.moves}
