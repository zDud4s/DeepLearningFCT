"""Naive DQN agent — no replay buffer, no target network"""
from __future__ import annotations

import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .network import SmallQNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BasicDQNAgent:
    """One gradient step per env step, bootstrapped from the same online net

    No replay buffer, no target net — the textbook starting point before adding
    the usual stabilisers
    """

    def __init__(self, state_shape: Tuple[int, int, int],
                 *,
                 n_actions: int = 2,
                 gamma: float = 0.99,
                 learning_rate: float = 1e-4,
                 grad_clip_norm: float = 10.0,
                 device: torch.device = DEVICE):
        channels, height, width = state_shape
        self.state_shape = state_shape
        self.n_actions = int(n_actions)
        self.gamma = float(gamma)
        self.grad_clip_norm = float(grad_clip_norm)
        self.device = device
        self.online = SmallQNetwork(channels, self.n_actions, height, width).to(device)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.online(state_t)
        return int(torch.argmax(q_values, dim=1).item())

    def max_q_values(self, state: np.ndarray) -> float:
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        with torch.no_grad():
            return float(self.online(state_t).max(dim=1).values.item())

    def train_step(self, state, action, reward, next_state, done) -> Dict[str, float]:
        state_t = torch.as_tensor(state, dtype=torch.float32,
                                  device=self.device).unsqueeze(0)
        next_state_t = torch.as_tensor(next_state, dtype=torch.float32,
                                       device=self.device).unsqueeze(0)
        action_t = torch.tensor([[int(action)]], dtype=torch.long, device=self.device)
        reward_t = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
        done_t = torch.tensor([float(done)], dtype=torch.float32, device=self.device)

        q_sa = self.online(state_t).gather(1, action_t).squeeze(1)
        with torch.no_grad():
            next_q = self.online(next_state_t).max(dim=1).values
            target = reward_t + self.gamma * (1.0 - done_t) * next_q

        loss = F.mse_loss(q_sa, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()
        td_error = target.detach() - q_sa.detach()
        return {
            "loss": float(loss.detach().cpu().item()),
            "q_value": float(q_sa.detach().cpu().item()),
            "target": float(target.detach().cpu().item()),
            "td_error": float(td_error.detach().cpu().item()),
        }

    def save(self, path: Path, *, cfg: Config | None = None,
             extras: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"model_state_dict": self.online.state_dict()}
        if cfg is not None:
            payload["config"] = asdict(cfg)
        if extras:
            payload.update(extras)
        torch.save(payload, path)
        return path

    @classmethod
    def from_checkpoint(cls, path: Path,
                        state_shape: Tuple[int, int, int],
                        *, gamma: float = 0.99,
                        learning_rate: float = 1e-4,
                        device: torch.device = DEVICE) -> "BasicDQNAgent":
        agent = cls(state_shape, gamma=gamma, learning_rate=learning_rate, device=device)
        try:
            checkpoint = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        agent.online.load_state_dict(state_dict)
        agent.online.eval()
        return agent
