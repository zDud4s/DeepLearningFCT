"""Open a pygame window and watch a policy play in real time"""
from __future__ import annotations

import time
from typing import Dict

from .config import Config
from .env import make_env, reset_env
from .policies import Policy


class LivePlayer:
    """Roll a policy on the pygame window — needs `pygame` installed"""

    def __init__(self, cfg: Config, *, fps: int = 10):
        self.cfg = cfg
        self.fps = max(1, int(fps))

    def play_episode(self, policy: Policy, *, difficulty: int = 0,
                     seed: int = 2026) -> Dict[str, float]:
        env = make_env(self.cfg, difficulty=difficulty,
                       include_semantic_info=policy.needs_semantic_info,
                       render_mode="human")
        env.metadata["render_fps"] = self.fps
        try:
            obs, info = reset_env(env, seed=seed)
            policy.reset()
            done = False
            steps = 0
            ep_return = 0.0
            while not done:
                action = policy.select_action(obs, info, env.action_space)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_return += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            return {
                "score": env.score,
                "collisions": env.collisions,
                "return": ep_return,
                "steps": steps,
            }
        finally:
            env.close()

    def play_many(self, policy: Policy, *, difficulty: int = 0,
                  seed: int = 2026, episodes: int = 1) -> list[Dict[str, float]]:
        results: list[Dict[str, float]] = []
        for ep in range(episodes):
            result = self.play_episode(policy, difficulty=difficulty,
                                       seed=seed + ep)
            results.append(result)
        return results
