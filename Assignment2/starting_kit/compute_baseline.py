"""
compute_baseline.py
====================
Compute baseline scores for random and always-up policies at each difficulty.

Usage:
    python compute_baseline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
from space_race_env import SpaceRaceEnv

N_EPISODES = 100

policies = {
    'random': lambda env: env.action_space.sample(),
    'always_up': lambda env: 0,
}

print("Baseline scores (N={} episodes each):".format(N_EPISODES))
print()

for policy_name, policy_fn in policies.items():
    print(f"{policy_name}:")
    for difficulty in range(4):
        scores = []
        for ep in range(N_EPISODES):
            env = SpaceRaceEnv(difficulty=difficulty, round_time_seconds=60, ticks_per_second=10)
            obs, _ = env.reset(seed=ep)
            done = False
            while not done:
                action = policy_fn(env)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
            scores.append(env.score)
            env.close()
        print(
            f"  difficulty={difficulty}  "
            f"mean={np.mean(scores):.2f}  "
            f"std={np.std(scores):.2f}  "
            f"min={min(scores)}  max={max(scores)}"
        )
    print()
