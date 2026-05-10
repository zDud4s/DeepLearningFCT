"""Run a `Policy` for N episodes and return aggregated metrics"""
from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

from .config import Config
from .env import make_env, reset_env
from .policies import Policy


def _score_from_info(info: dict, fallback: float) -> float:
    for key in ("score", "completed_runs", "crossings", "top_crossings"):
        if key in info:
            try:
                return float(info[key])
            except (TypeError, ValueError):
                pass
    return float(fallback)


class Evaluator:
    """Mirrors the Codabench eval setup: 10 episodes, seeds 2026..2035 (via
    `cfg.base_eval_seed`), `include_semantic_info=False`
    """

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def run(self, policy: Policy,
            *, difficulty: Optional[int] = None,
            n_episodes: Optional[int] = None,
            base_seed: Optional[int] = None,
            include_semantic_info: Optional[bool] = None) -> Dict[str, object]:
        cfg = self.cfg
        n_ep = cfg.eval_episodes if n_episodes is None else int(n_episodes)
        diff = cfg.difficulty if difficulty is None else int(difficulty)
        base = cfg.base_eval_seed if base_seed is None else int(base_seed)
        semantic = (policy.needs_semantic_info
                    if include_semantic_info is None else bool(include_semantic_info))

        env = make_env(cfg, difficulty=diff, include_semantic_info=semantic)
        scores: List[float] = []
        returns: List[float] = []
        collisions: List[float] = []
        steps: List[int] = []
        for ep in range(n_ep):
            seed = base + ep
            obs, info = reset_env(env, seed=seed)
            policy.reset()
            done = False
            ep_return = 0.0
            n_steps = 0
            while not done and n_steps < cfg.max_steps_per_episode:
                action = policy.select_action(obs, info, env.action_space)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_return += float(reward)
                n_steps += 1
                done = bool(terminated or truncated)
            scores.append(_score_from_info(info, fallback=ep_return))
            returns.append(ep_return)
            collisions.append(float(info.get("collisions", math.nan)))
            steps.append(n_steps)
        env.close()

        return {
            "policy": getattr(policy, "name", policy.__class__.__name__),
            "difficulty": int(diff),
            "episodes": int(n_ep),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "min_score": float(np.min(scores)),
            "max_score": float(np.max(scores)),
            "mean_return": float(np.mean(returns)),
            "mean_collisions": float(np.nanmean(collisions)),
            "mean_steps": float(np.mean(steps)),
            "scores": [float(s) for s in scores],
        }
