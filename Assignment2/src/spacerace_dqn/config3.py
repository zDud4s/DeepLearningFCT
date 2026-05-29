"""Task 3 run configuration — adds decay schedules for exploration."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass
class Config3:
    # --- reproducibility & env ---
    seed:                      int   = 7
    difficulty:                int   = 0
    obs_mode:                  str   = "rgb"
    include_semantic_info:     bool  = False
    round_time_seconds:        int   = 60
    ticks_per_second:          int   = 10
    max_steps_per_episode:     int   = 600
    frame_stack:               int   = 1

    # --- network / optimiser ---
    gamma:                     float = 0.99
    learning_rate:             float = 1e-4
    grad_clip_norm:            float = 10.0

    # --- replay buffer ---
    buffer_capacity:           int   = 10_000
    batch_size:                int   = 64
    warmup_steps:              int   = 1_000
    use_per:                   bool  = False
    per_alpha:                 float = 0.6

    # --- target network ---
    target_update_freq:        int   = 100

    # --- training schedule ---
    episodes:                  int   = 300      # Enough to see convergence differences
    eval_every:                int   = 20
    eval_episodes:             int   = 5        # Faster eval for testing
    baseline_episodes:         int   = 5
    base_eval_seed:            int   = 2026

    # --- exploration (Task 3 Additions) ---
    exploration:               str   = "epsilon"   # "epsilon" | "boltzmann"
    decay_schedule:            str   = "linear"    # "linear" | "exponential"

    # ε-greedy
    epsilon_start:             float = 1.0
    epsilon_end:               float = 0.05
    epsilon_decay_steps:       int   = 50_000

    # Boltzmann
    temperature_start:         float = 5.0
    temperature_end:           float = 0.2
    temperature_decay_steps:   int   = 50_000

    # --- heuristic warm-start ---
    heuristic_warmup_episodes: int   = 10

    # --- output ---
    submission_name:           str   = "task3_exploration"
    checkpoint_name:           str   = "task3_exploration.pt"