"""Runs one Task 1 training run end-to-end"""
from __future__ import annotations

import math
import random
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from .agent import DEVICE, BasicDQNAgent
from .config import Config
from .env import make_env, reset_env
from .evaluator import Evaluator
from .preprocessing import FrameStacker


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def epsilon_by_step(step: int, cfg: Config) -> float:
    fraction = min(1.0, step / max(1, cfg.epsilon_decay_steps))
    return cfg.epsilon_start + fraction * (cfg.epsilon_end - cfg.epsilon_start)


def safe_rate(value: float, elapsed_minutes: float) -> float:
    if not np.isfinite(value) or elapsed_minutes <= 0:
        return float("nan")
    return float(value / elapsed_minutes)


class DQNTrainer:
    """Owns the training loop, periodic evaluation and run history"""

    def __init__(self, cfg: Config, *, device: torch.device = DEVICE,
                 verbose: bool = True):
        self.cfg = cfg
        self.device = device
        self.verbose = verbose
        self.agent: Optional[BasicDQNAgent] = None
        self.history: List[Dict] = []
        self.eval_history: List[Dict] = []
        self.train_minutes: float = 0.0
        self.state_shape: Optional[Tuple[int, int, int]] = None

    def train(self) -> BasicDQNAgent:
        cfg = self.cfg
        set_seed(cfg.seed)
        env = make_env(cfg, include_semantic_info=cfg.include_semantic_info)
        sample_obs, _ = reset_env(env, seed=cfg.seed)
        sample_state = FrameStacker(cfg.frame_stack).reset(sample_obs)
        self.state_shape = tuple(sample_state.shape)
        self.agent = BasicDQNAgent(
            state_shape=self.state_shape,
            n_actions=int(env.action_space.n),
            gamma=cfg.gamma,
            learning_rate=cfg.learning_rate,
            grad_clip_norm=cfg.grad_clip_norm,
            device=self.device,
        )
        if self.verbose:
            params = sum(p.numel() for p in self.agent.online.parameters())
            print(f"[train] device={self.device} state_shape={self.state_shape} "
                  f"params={params}")

        global_step = 0
        start = time.time()
        evaluator = Evaluator(cfg)
        for episode in range(1, cfg.episodes + 1):
            result, global_step = self._run_episode(env, train=True,
                                                    global_step=global_step,
                                                    seed=cfg.seed + episode)
            elapsed = (time.time() - start) / 60.0
            result.update({
                "episode": int(episode),
                "global_step": int(global_step),
                "elapsed_minutes": float(elapsed),
                "score_per_minute": safe_rate(result["score"], elapsed),
                "q_value_per_minute": safe_rate(result["q_value"], elapsed),
            })
            self.history.append(result)

            if episode % cfg.eval_every == 0 or episode == 1:
                eval_result = self._evaluate_with_q(evaluator, cfg.difficulty)
                eval_result.update({
                    "episode": int(episode),
                    "global_step": int(global_step),
                    "elapsed_minutes": float(elapsed),
                    "mean_score_per_minute": safe_rate(eval_result["mean_score"], elapsed),
                    "q_per_minute": safe_rate(eval_result["mean_max_q"], elapsed),
                })
                self.eval_history.append(eval_result)
                if self.verbose:
                    print(f"  ep={episode:04d} step={global_step:06d} "
                          f"train_score={result['score']:.2f} "
                          f"eval_mean={eval_result['mean_score']:.2f} "
                          f"loss={result['loss']:.4f} q={result['q_value']:.3f} "
                          f"eps={result['epsilon']:.3f} elapsed={elapsed:.1f}m")

        env.close()
        self.train_minutes = (time.time() - start) / 60.0
        return self.agent

    def final_eval(self, *, difficulty: Optional[int] = None,
                   n_episodes: Optional[int] = None) -> Dict[str, object]:
        if self.agent is None:
            raise RuntimeError("Call train() before final_eval().")
        evaluator = Evaluator(self.cfg)
        return self._evaluate_with_q(evaluator,
                                     self.cfg.difficulty if difficulty is None else difficulty,
                                     n_episodes=n_episodes)

    def _run_episode(self, env, *, train: bool, global_step: int = 0,
                     seed: Optional[int] = None) -> Tuple[Dict[str, float], int]:
        cfg = self.cfg
        agent = self.agent
        obs, info = reset_env(env, seed=seed)
        stacker = FrameStacker(cfg.frame_stack)
        state = stacker.reset(obs)
        done = False
        steps = 0
        ep_return = 0.0
        losses: List[float] = []
        q_values: List[float] = []
        epsilons: List[float] = []
        actions: List[int] = []

        while not done and steps < cfg.max_steps_per_episode:
            epsilon = epsilon_by_step(global_step, cfg) if train else 0.0
            action = agent.select_action(state, epsilon=epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            next_state = stacker.append(next_obs)
            if train:
                stats = agent.train_step(state, action, reward, next_state, done)
                losses.append(stats["loss"])
                q_values.append(stats["q_value"])
                epsilons.append(epsilon)
                global_step += 1
            actions.append(int(action))
            state = next_state
            ep_return += float(reward)
            steps += 1

        return {
            "score": float(getattr(env, "score", ep_return)),
            "return": float(ep_return),
            "collisions": float(getattr(env, "collisions", float("nan"))),
            "steps": int(steps),
            "loss": float(np.mean(losses)) if losses else float("nan"),
            "q_value": float(np.mean(q_values)) if q_values else float("nan"),
            "epsilon": float(np.mean(epsilons)) if epsilons else 0.0,
            "action_up_ratio": (float(np.mean([1.0 if a == 0 else 0.0 for a in actions]))
                                if actions else float("nan")),
        }, global_step

    def _evaluate_with_q(self, evaluator: Evaluator, difficulty: int,
                         *, n_episodes: Optional[int] = None) -> Dict[str, object]:
        """Greedy eval episodes plus mean max-Q sampled on the same seeds"""
        cfg = self.cfg
        n_ep = cfg.eval_episodes if n_episodes is None else int(n_episodes)
        from .policies import Policy

        class _GreedyDQNPolicy(Policy):
            name = "dqn_greedy_inproc"

            def __init__(inner_self, agent: BasicDQNAgent, stack: int):
                inner_self.agent = agent
                inner_self.stack_size = stack
                inner_self.stacker: Optional[FrameStacker] = None

            def reset(inner_self) -> None:
                inner_self.stacker = None

            def select_action(inner_self, obs, info, action_space) -> int:
                if inner_self.stacker is None:
                    inner_self.stacker = FrameStacker(inner_self.stack_size)
                    state = inner_self.stacker.reset(obs)
                else:
                    state = inner_self.stacker.append(obs)
                return inner_self.agent.select_action(state, epsilon=0.0)

        policy = _GreedyDQNPolicy(self.agent, cfg.frame_stack)
        result = evaluator.run(policy, difficulty=difficulty, n_episodes=n_ep,
                               include_semantic_info=False)
        result["mean_max_q"] = self._sample_mean_max_q(difficulty=difficulty)
        return result

    def _sample_mean_max_q(self, *, difficulty: int, sample_steps: int = 60,
                           n_seeds: Optional[int] = None) -> float:
        cfg = self.cfg
        n = cfg.eval_episodes if n_seeds is None else n_seeds
        env = make_env(cfg, difficulty=difficulty, include_semantic_info=False)
        all_qs: List[float] = []
        for idx in range(n):
            obs, _ = reset_env(env, seed=cfg.base_eval_seed + idx)
            stacker = FrameStacker(cfg.frame_stack)
            state = stacker.reset(obs)
            for _ in range(sample_steps):
                q = self.agent.max_q_values(state)
                all_qs.append(q)
                action = self.agent.select_action(state, epsilon=0.0)
                next_obs, _, terminated, truncated, _ = env.step(action)
                state = stacker.append(next_obs)
                if terminated or truncated:
                    break
        env.close()
        return float(np.mean(all_qs)) if all_qs else float("nan")
