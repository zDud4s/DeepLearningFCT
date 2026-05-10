"""
local_eval.py
==============
Simulates exactly what the Codabench evaluation worker will do.
Run this before submitting to catch bugs early.

Usage:
    python local_eval.py [--difficulty 0]
"""

import argparse
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from space_race_env import SpaceRaceEnv
from agent import Agent

N_EPISODES = 10
BASE_SEED  = 2026


def evaluate(difficulty: int) -> None:
    agent = Agent()
    scores = []

    for ep in range(N_EPISODES):
        # Mirrors Codabench evaluation: RGB observations, no semantic info
        env = SpaceRaceEnv(
            difficulty=difficulty,
            round_time_seconds=60,
            ticks_per_second=10,
            obs_mode="rgb",
            include_semantic_info=False,  # Same as Codabench evaluation
        )
        obs, _ = env.reset(seed=BASE_SEED + ep)
        done = False
        while not done:
            action = agent.select_action(obs)
            action = int(np.clip(int(action), 0, 1))
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
        scores.append(env.score)
        env.close()
        print(f"  Episode {ep + 1:2d}/{N_EPISODES}  ->  score = {env.score}")

    print(
        f"\n--- Results (difficulty={difficulty}) ---\n"
        f"  Mean : {np.mean(scores):.2f}\n"
        f"  Std  : {np.std(scores):.2f}\n"
        f"  Min  : {min(scores)}\n"
        f"  Max  : {max(scores)}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--difficulty", type=int, default=0, choices=[0, 1, 2, 3])
    args = parser.parse_args()
    evaluate(args.difficulty)
