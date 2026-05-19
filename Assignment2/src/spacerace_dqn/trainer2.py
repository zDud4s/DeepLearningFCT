from __future__ import annotations

import math
import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent2 import DEVICE, DQNAgent
from .config2 import Config2
from .env import make_env, reset_env
from .evaluator import Evaluator
from .preprocessing import FrameStacker


# Utilities
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, cfg: Config2) -> float:
    frac = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + frac * (cfg.epsilon_end - cfg.epsilon_start)


def temperature_by_step(step: int, cfg: Config2) -> float:
    frac = min(1.0, step / max(1, cfg.temperature_decay_steps))
    return cfg.temperature_start + frac * (cfg.temperature_end - cfg.temperature_start)


def safe_rate(value: float, elapsed_minutes: float) -> float:
    if not np.isfinite(value) or elapsed_minutes <= 0:
        return float("nan")
    return float(value / elapsed_minutes)


def _best_heuristic_policy():
    """Best available heuristic with graceful fallback chain."""
    for cls_name in ("HeuristicV26MirrorEnv", "HeuristicV18FullTreeS3",
                     "HeuristicV15FullTree"):
        try:
            from . import improved_heuristics as ih
            cls = getattr(ih, cls_name)
            try:
                return cls(k=4)
            except TypeError:
                return cls()
        except (ImportError, AttributeError):
            continue
    # Final fallback: built-in semantic heuristic (needs semantic env)
    from .policies import SemanticHeuristicPolicy
    return SemanticHeuristicPolicy()



# Trainer
class DQNTrainer2:
    """End-to-end training loop for the Task 2 enhanced DQN."""

    def __init__(self, cfg: Config2, *,
                 device: torch.device = DEVICE,
                 verbose: bool = True):
        self.cfg           = cfg
        self.device        = device
        self.verbose       = verbose
        self.agent:        Optional[DQNAgent] = None
        self.history:      List[Dict]         = []
        self.eval_history: List[Dict]         = []
        self.train_minutes: float             = 0.0
        self.state_shape:  Optional[Tuple]    = None


    # Public API
    def train(self) -> DQNAgent:
        cfg = self.cfg
        set_seed(cfg.seed)

        # training env never has semantic info
        env = make_env(cfg, include_semantic_info=False)
        sample_obs, _ = reset_env(env, seed=cfg.seed)
        self.state_shape = tuple(FrameStacker(cfg.frame_stack).reset(sample_obs).shape)

        self.agent = DQNAgent(
            state_shape        = self.state_shape,
            n_actions          = int(env.action_space.n),
            gamma              = cfg.gamma,
            learning_rate      = cfg.learning_rate,
            grad_clip_norm     = cfg.grad_clip_norm,
            buffer_capacity    = cfg.buffer_capacity,
            batch_size         = cfg.batch_size,
            use_per            = cfg.use_per,
            per_alpha          = cfg.per_alpha,
            target_update_freq = cfg.target_update_freq,
            warmup_steps       = cfg.warmup_steps,
            device             = self.device,
            seed               = cfg.seed,
        )

        if self.verbose:
            params   = sum(p.numel() for p in self.agent.online.parameters())
            buf_type = "PER" if cfg.use_per else "Uniform"
            print(
                f"[DQNTrainer2]  device={self.device}  state={self.state_shape}  "
                f"params={params:,}\n"
                f"  buffer={buf_type} cap={cfg.buffer_capacity:,}  "
                f"batch={cfg.batch_size}  warmup={cfg.warmup_steps:,}  "
                f"target_update={cfg.target_update_freq}  "
                f"exploration={cfg.exploration}  semantic=False"
            )

        start       = time.time()
        evaluator   = Evaluator(cfg)
        global_step = 0

        # Phase 0 — heuristic warm-start
        # uses its own semantic env, does NOT pass the RGB training env
        if cfg.heuristic_warmup_episodes > 0:
            global_step = self._heuristic_warmup(cfg.heuristic_warmup_episodes)
            if self.verbose:
                elapsed = (time.time() - start) / 60.0
                print(f"  [warmup]  {len(self.agent.buffer):,} transitions "
                      f"from {cfg.heuristic_warmup_episodes} heuristic episodes "
                      f"({elapsed:.1f} min)")

        # Phase 1 — main training loop (RGB only, no semantic)
        for episode in range(1, cfg.episodes + 1):
            result, global_step = self._run_episode(
                env, train=True, global_step=global_step,
                seed=cfg.seed + cfg.heuristic_warmup_episodes + episode,
            )
            elapsed = (time.time() - start) / 60.0
            result.update({
                "episode":          int(episode),
                "global_step":      int(global_step),
                "elapsed_minutes":  float(elapsed),
                "score_per_minute": safe_rate(result["score"], elapsed),
                "buffer_size":      int(len(self.agent.buffer)),
            })
            self.history.append(result)

            if episode % cfg.eval_every == 0 or episode == 1:
                ev = self._evaluate_with_q(evaluator, cfg.difficulty)
                ev.update({
                    "episode":         int(episode),
                    "global_step":     int(global_step),
                    "elapsed_minutes": float(elapsed),
                })
                self.eval_history.append(ev)
                if self.verbose:
                    print(
                        f"  ep={episode:04d}  step={global_step:06d}  "
                        f"buf={len(self.agent.buffer):,}  "
                        f"score={result['score']:.1f}  "
                        f"eval={ev['mean_score']:.2f}  "
                        f"loss={result['loss']:.4f}  "
                        f"q={result['q_value']:.3f}  "
                        f"explore={result['explore_param']:.3f}  "
                        f"{elapsed:.1f}m"
                    )

        env.close()
        self.train_minutes = (time.time() - start) / 60.0
        return self.agent

    def final_eval(self, *, difficulty: Optional[int] = None,
                   n_episodes: Optional[int] = None) -> Dict:
        if self.agent is None:
            raise RuntimeError("Call train() first.")
        return self._evaluate_with_q(
            Evaluator(self.cfg),
            self.cfg.difficulty if difficulty is None else difficulty,
            n_episodes=n_episodes,
        )

    # heuristic warmup with its own semantic env
    def _heuristic_warmup(self, n_episodes: int) -> int:
        """Fill the replay buffer with heuristic experience.

        Creates a SEPARATE env with include_semantic_info=True so the
        heuristic policy can read info["semantic_obs"].  The RGB obs
        (not semantic) is stored in the buffer — same format as training.
        """
        cfg     = self.cfg
        # Semantic env — only used here, closed when done
        sem_env = make_env(cfg, include_semantic_info=True)
        policy  = _best_heuristic_policy()
        stacker = FrameStacker(cfg.frame_stack)
        total   = 0

        for ep in range(n_episodes):
            obs, info = reset_env(sem_env, seed=cfg.seed + ep)
            policy.reset()
            state = stacker.reset(obs)   # obs is RGB (obs_mode="rgb")
            done  = False
            steps = 0
            while not done and steps < cfg.max_steps_per_episode:
                # Heuristic reads info["semantic_obs"] — works because sem_env
                action = policy.select_action(obs, info, sem_env.action_space)
                next_obs, reward, terminated, truncated, info = sem_env.step(int(action))
                done       = bool(terminated or truncated)
                next_state = stacker.append(next_obs)
                # Store RGB transition — matches what the DQN will train on
                self.agent.push(state, action, reward, next_state, done)
                obs, state = next_obs, next_state
                steps += 1
                total += 1

        sem_env.close()
        return total


    # Single episode
    def _run_episode(self, env, *, train: bool,
                     global_step: int = 0,
                     seed: Optional[int] = None) -> Tuple[Dict, int]:
        cfg   = self.cfg
        agent = self.agent

        obs, _    = reset_env(env, seed=seed)
        stacker   = FrameStacker(cfg.frame_stack)
        state     = stacker.reset(obs)
        done      = False
        steps     = 0
        ep_return = 0.0
        losses: List[float] = []
        q_vals: List[float] = []
        explores: List[float] = []
        actions: List[int] = []

        while not done and steps < cfg.max_steps_per_episode:
            if train:
                if cfg.exploration == "boltzmann":
                    temp   = temperature_by_step(global_step, cfg)
                    action = agent.select_action_boltzmann(state, temperature=temp)
                    ep_val = temp
                else:
                    ep_val = epsilon_by_step(global_step, cfg)
                    action = agent.select_action(state, epsilon=ep_val)
                explores.append(ep_val)
            else:
                action = agent.select_action(state, epsilon=0.0)

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done       = bool(terminated or truncated)
            next_state = stacker.append(next_obs)

            if train:
                agent.push(state, action, reward, next_state, done)
                stats = agent.train_step()
                if math.isfinite(stats["loss"]):
                    losses.append(stats["loss"])
                    q_vals.append(stats["q_value"])
                global_step += 1

            actions.append(int(action))
            state      = next_state
            obs        = next_obs
            ep_return += float(reward)
            steps     += 1

        explore_val = float(np.mean(explores)) if explores else 0.0
        return {
            "score":          float(getattr(env, "score", ep_return)),
            "return":         float(ep_return),
            "collisions":     float(getattr(env, "collisions", float("nan"))),
            "steps":          int(steps),
            "loss":           float(np.mean(losses))   if losses   else float("nan"),
            "q_value":        float(np.mean(q_vals))   if q_vals   else float("nan"),
            "explore_param":  explore_val,
            "epsilon":        explore_val,
            "action_up_ratio": (float(np.mean([1 if a == 0 else 0 for a in actions]))
                                if actions else float("nan")),
            "grad_steps":     int(agent.grad_steps),
        }, global_step


    # Evaluation
    def _evaluate_with_q(self, evaluator: Evaluator, difficulty: int,
                         *, n_episodes: Optional[int] = None) -> Dict:
        cfg  = self.cfg
        n_ep = cfg.eval_episodes if n_episodes is None else int(n_episodes)

        from .policies import Policy

        class _GreedyPolicy(Policy):
            name = "dqn2_greedy_inproc"
            def __init__(inner, agent: DQNAgent, stack: int):
                inner.agent = agent; inner.stack_size = stack
                inner.stacker: Optional[FrameStacker] = None
            def reset(inner) -> None:
                inner.stacker = None
            def select_action(inner, obs, info, action_space) -> int:
                if inner.stacker is None:
                    inner.stacker = FrameStacker(inner.stack_size)
                    s = inner.stacker.reset(obs)
                else:
                    s = inner.stacker.append(obs)
                return inner.agent.select_action(s, epsilon=0.0)

        policy = _GreedyPolicy(self.agent, cfg.frame_stack)
        result = evaluator.run(policy, difficulty=difficulty,
                               n_episodes=n_ep, include_semantic_info=False)
        result["mean_max_q"] = self._sample_mean_max_q(difficulty=difficulty)
        return result

    def _sample_mean_max_q(self, *, difficulty: int,
                           sample_steps: int = 60,
                           n_seeds: Optional[int] = None) -> float:
        cfg = self.cfg
        n   = cfg.eval_episodes if n_seeds is None else n_seeds
        env = make_env(cfg, difficulty=difficulty, include_semantic_info=False)
        qs: List[float] = []
        for idx in range(n):
            obs, _ = reset_env(env, seed=cfg.base_eval_seed + idx)
            stacker = FrameStacker(cfg.frame_stack)
            state   = stacker.reset(obs)
            for _ in range(sample_steps):
                qs.append(self.agent.max_q_values(state))
                action = self.agent.select_action(state, epsilon=0.0)
                next_obs, _, terminated, truncated, _ = env.step(action)
                state = stacker.append(next_obs)
                if terminated or truncated:
                    break
        env.close()
        return float(np.mean(qs)) if qs else float("nan")