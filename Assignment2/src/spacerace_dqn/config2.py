"""Task 2 run configuration — extends the Task 1 Config with replay buffer,
target network, and Boltzmann exploration parameters.

Every field has a default that reproduces the standard Task 2 setup without
any manual tuning, making experiments fully reproducible by recording config.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Config2:
    """Hyperparameters for one enhanced DQN run.

    Exploration:
    exploration = "epsilon":
        Standard ε-greedy with linear decay from epsilon_start to
        epsilon_end over epsilon_decay_steps environment steps.
    exploration = "boltzmann":
        Softmax over Q-values with temperature T decaying linearly from
        temperature_start to temperature_end.
        
    ----------------------------------

    Heuristic warm-start:
    heuristic_warmup_episodes > 0:
        Play that many episodes with the best available heuristic policy
        and push transitions into the replay buffer before gradient descent
        begins.  Set to 0 to disable.
            
    ----------------------------------

    IMPORTANT — include_semantic_info:
    Must be False so the training environment matches Codabench evaluation
    conditions exactly.  The agent's select_action(obs) receives only the
    RGB frame, just as it will on the leaderboard.  The heuristic warm-start
    uses a separate env with semantic info enabled only for the heuristic
    policy itself — the transitions pushed into the replay buffer still use
    RGB observations so there is no mismatch.
    """

    # --- reproducibility ---
    seed:                      int   = 7
    difficulty:                int   = 0
    obs_mode:                  str   = "rgb"
    include_semantic_info:     bool  = False   # MUST be False — matches Codabench
    round_time_seconds:        int   = 60
    ticks_per_second:          int   = 10
    max_steps_per_episode:     int   = 600

    # --- preprocessing ---
    frame_stack:               int   = 1

    # --- network / optimiser ---
    gamma:                     float = 0.99
    learning_rate:             float = 1e-4
    grad_clip_norm:            float = 10.0

    # --- replay buffer (§2.1) ---
    buffer_capacity:           int   = 10_000   # must be <= 10 000 per spec
    batch_size:                int   = 64
    warmup_steps:              int   = 1_000    # fill buffer this far before training
    use_per:                   bool  = False    # True -> PrioritizedReplayBuffer
    per_alpha:                 float = 0.6

    # --- target network (§2.3) ---
    target_update_freq:        int   = 500      # gradient steps between hard copies

    # --- training schedule ---
    episodes:                  int   = 150
    eval_every:                int   = 10
    eval_episodes:             int   = 10
    baseline_episodes:         int   = 10
    base_eval_seed:            int   = 2026

    # --- exploration ---
    exploration:               str   = "epsilon"   # "epsilon" | "boltzmann"

    # ε-greedy
    epsilon_start:             float = 1.0
    epsilon_end:               float = 0.05
    epsilon_decay_steps:       int   = 40_000

    # Boltzmann
    temperature_start:         float = 5.0
    temperature_end:           float = 0.2
    temperature_decay_steps:   int   = 40_000

    # --- heuristic warm-start ---
    heuristic_warmup_episodes: int   = 10

    # --- output ---
    submission_name:           str   = "task2_enhanced_dqn"
    checkpoint_name:           str   = "task2_enhanced_dqn.pt"