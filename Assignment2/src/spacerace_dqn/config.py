"""Run configuration for SpaceRace DQN experiments"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Config:
    """Hyperparameters for one training/eval run

    Defaults reproduce the Task 1 reference run: seed 7, difficulty 0,
    no frame stacking, gamma 0.99, lr 1e-4, epsilon 1.0 -> 0.05 over
    30k env steps, 120 episodes
    """

    seed: int = 7
    difficulty: int = 0
    obs_mode: str = "rgb"
    include_semantic_info: bool = True
    round_time_seconds: int = 60
    ticks_per_second: int = 10
    max_steps_per_episode: int = 600

    frame_stack: int = 1

    gamma: float = 0.99
    learning_rate: float = 1e-4
    grad_clip_norm: float = 10.0

    episodes: int = 120
    eval_every: int = 10
    eval_episodes: int = 10
    baseline_episodes: int = 10
    base_eval_seed: int = 2026  # same seed Codabench uses in starting_kit/local_eval.py

    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 30_000

    submission_name: str = "task1_basic_dqn"
    checkpoint_name: str = "task1_basic_dqn.pt"
