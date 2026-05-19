"""Task 2 — Enhanced DQN agent
   Replay buffer + target network + Boltzmann exploration + optional PER

Public additions over BasicDQNAgent (Task 1):
  ReplayBuffer               — uniform FIFO circular buffer (§2.1)
  PrioritizedReplayBuffer    — PER: high-TD-error transitions sampled more often
  DQNAgent                   — online net + hard-updated target net + batch steps
"""
from __future__ import annotations

import random
from collections import deque
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config
from .network import SmallQNetwork

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Transition container
class Transition(NamedTuple):
    state:      np.ndarray
    action:     int
    reward:     float
    next_state: np.ndarray
    done:       float   # stored as 0.0 / 1.0 for vectorised arithmetic


# 2.1  Uniform Replay Buffer
class ReplayBuffer:
    """Fixed-size circular (FIFO) buffer with uniform random sampling.

    When the buffer is full, the oldest transition is silently overwritten —
    implemented for free by ``collections.deque(maxlen=capacity)``.

    Args:
        capacity : Maximum number of transitions stored.
        seed     : Optional RNG seed for reproducible sampling.
    """

    def __init__(self, capacity: int = 10_000, seed: Optional[int] = None):
        self.capacity = int(capacity)
        self._buf: deque[Transition] = deque(maxlen=self.capacity)
        self._rng = random.Random(seed)

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Store one transition (matches assignment base-class method name)."""
        self._buf.append(Transition(
            state=state,
            action=int(action),
            reward=float(reward),
            next_state=next_state,
            done=float(done),
        ))

    push = add   # alias so DQNAgent can call .push() as well

    def sample(self, batch_size: int) -> List[Transition]:
        """Sample ``batch_size`` transitions uniformly with replacement."""
        return self._rng.choices(self._buf, k=batch_size)

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return f"ReplayBuffer(capacity={self.capacity}, size={len(self)})"


# 2.1  Prioritized Experience Replay
class PrioritizedReplayBuffer:
    """Proportional PER (Schaul et al., 2016).

    Sampling probability:

        P(i) = |delta_i|^alpha / sum_j |delta_j|^alpha

    where ``delta_i`` is the TD error and ``alpha`` controls prioritization
    strength (alpha=0 → uniform, alpha=1 → full priority).

    Args:
        capacity         : Max buffer size (keep <= 10 000 per assignment spec).
        alpha            : Priority exponent.
        default_priority : Priority assigned to brand-new transitions;
                           set high so every new sample is seen at least once.
        seed             : RNG seed.
    """

    def __init__(self, capacity: int = 10_000, alpha: float = 0.6,
                 default_priority: float = 1.0, seed: Optional[int] = None):
        self.capacity         = int(capacity)
        self.alpha            = float(alpha)
        self.default_priority = float(default_priority)
        self._buf:        List[Transition] = []
        self._priorities: List[float]      = []
        self._pos = 0
        self._rng = np.random.default_rng(seed)

    def add(self, state, action, reward, next_state, done,
            priority: Optional[float] = None) -> None:
        t = Transition(state, int(action), float(reward), next_state, float(done))
        p = float(priority) if priority is not None else self.default_priority
        if len(self._buf) < self.capacity:
            self._buf.append(t)
            self._priorities.append(p)
        else:
            self._buf[self._pos]        = t
            self._priorities[self._pos] = p
        self._pos = (self._pos + 1) % self.capacity

    push = add

    def sample(self, batch_size: int
               ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Return (transitions, indices, importance_weights).

        Indices are returned so callers can update priorities after each step.
        """
        priorities = np.array(self._priorities, dtype=np.float64)
        probs      = priorities ** self.alpha
        probs     /= probs.sum()
        indices    = self._rng.choice(len(self._buf), size=batch_size,
                                      replace=True, p=probs)
        transitions = [self._buf[i] for i in indices]
        weights     = (len(self._buf) * probs[indices]) ** -1.0
        weights    /= weights.max()
        return transitions, indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        """Refresh stored priorities with fresh TD errors (call after train_step)."""
        for idx, err in zip(indices, td_errors):
            self._priorities[idx] = float(abs(err)) + 1e-6

    def __len__(self) -> int:
        return len(self._buf)

    def __repr__(self) -> str:
        return (f"PrioritizedReplayBuffer(capacity={self.capacity}, "
                f"alpha={self.alpha}, size={len(self)})")



# 2.2 / 2.3  DQN Agent
class DQNAgent:
    """DQN with experience replay and a hard-updated target network.

    Design rationale
    Replay buffer (2.1 / 2.2)
        Transitions are stored in a ReplayBuffer (or
        PrioritizedReplayBuffer).  Each train_step draws a
        mini-batch of batch_size transitions.  Benefits:
          - Breaks temporal correlation between consecutive samples.
          - Each transition contributes to ~capacity/batch_size gradient
            updates (better sample efficiency than online 1-step updates).
          - Off-policy Q-learning can exploit replayed transitions
            regardless of which policy collected them.

    Target network (2.3)
        A frozen copy of the online network (self.target) provides the
        bootstrapped Q-value in the TD target::

            target = r + gamma * (1 - done) * max_{a'} Q_target(s', a')

        Weights are hard-copied from self.online every
        target_update_freq gradient steps.  This removes the positive
        feedback loop present in the basic DQN where both the prediction
        and the target shift after every weight update.

    Exploration
        - select_action(state, epsilon)          — standard ε-greedy
        - select_action_boltzmann(state, temp)  — softmax over Q-values

    The public API is identical to BasicDQNAgent so Evaluator and
    SubmissionPackager require no changes.
    """

    def __init__(
        self,
        state_shape:        Tuple[int, int, int],
        *,
        n_actions:          int   = 2,
        gamma:              float = 0.99,
        learning_rate:      float = 1e-4,
        grad_clip_norm:     float = 10.0,
        # replay buffer
        buffer_capacity:    int   = 10_000,
        batch_size:         int   = 64,
        use_per:            bool  = False,
        per_alpha:          float = 0.6,
        # target network
        target_update_freq: int   = 500,
        warmup_steps:       int   = 1_000,
        # misc
        device:             torch.device = DEVICE,
        seed:               Optional[int] = None,
    ):
        channels, height, width = state_shape
        self.state_shape        = state_shape
        self.n_actions          = int(n_actions)
        self.gamma              = float(gamma)
        self.grad_clip_norm     = float(grad_clip_norm)
        self.batch_size         = int(batch_size)
        self.target_update_freq = int(target_update_freq)
        self.warmup_steps       = int(warmup_steps)
        self.use_per            = bool(use_per)
        self.device             = device

        # Networks
        self.online = SmallQNetwork(channels, self.n_actions, height, width).to(device)
        self.target = SmallQNetwork(channels, self.n_actions, height, width).to(device)
        self._hard_update_target()
        self.target.eval()   # target is never updated by backprop

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)

        # Replay buffer
        if use_per:
            self.buffer: ReplayBuffer | PrioritizedReplayBuffer = \
                PrioritizedReplayBuffer(capacity=buffer_capacity,
                                        alpha=per_alpha, seed=seed)
        else:
            self.buffer = ReplayBuffer(capacity=buffer_capacity, seed=seed)

        # Counters
        self.grad_steps = 0
        self.env_steps  = 0


    # Target network helpers
    def _hard_update_target(self) -> None:
        """Copy all online weights into target (§2.3 hard update)."""
        self.target.load_state_dict(self.online.state_dict())

    def _soft_update_target(self, tau: float = 0.01) -> None:
        """Polyak-average online → target (§2.3 soft update, alternative).

        target_w = tau * online_w + (1 - tau) * target_w
        """
        for t_p, o_p in zip(self.target.parameters(), self.online.parameters()):
            t_p.data.copy_(tau * o_p.data + (1.0 - tau) * t_p.data)


    # Tensor helper
    def _t(self, x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)


    # Action selection
    def select_action(self, state: np.ndarray, epsilon: float = 0.0) -> int:
        """ε-greedy: uniform random with prob ε, otherwise argmax Q."""
        if random.random() < epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            q = self.online(self._t(state).unsqueeze(0))
        return int(torch.argmax(q, dim=1).item())

    def select_action_boltzmann(self, state: np.ndarray,
                                temperature: float = 1.0) -> int:
        """Boltzmann (softmax) exploration.

        P(a_i) = exp(Q(a_i) / T) / sum_j exp(Q(a_j) / T)

        High T  — near-uniform (explore).
        T -> 0  — argmax (exploit).

        Unlike ε-greedy, even while exploring the agent still favours
        actions with higher Q-values under its current estimates.
        """
        with torch.no_grad():
            q = self.online(self._t(state).unsqueeze(0)).squeeze(0)
        probs = F.softmax(q / max(float(temperature), 1e-6), dim=0).cpu().numpy()
        return int(np.random.choice(self.n_actions, p=probs))

    def max_q_values(self, state: np.ndarray) -> float:
        with torch.no_grad():
            return float(self.online(self._t(state).unsqueeze(0)).max().item())


    # Buffer interaction
    def push(self, state, action, reward, next_state, done) -> None:
        """Store one transition.  Alias: .add() for assignment compatibility."""
        self.buffer.add(state, action, reward, next_state, done)
        self.env_steps += 1

    add = push

    # Training step  (2.2 + 2.3)
    def train_step(self, state=None, action=None, reward=None,
                   next_state=None, done=None) -> Dict[str, float]:
        """Draw a mini-batch and take one gradient step.

        Positional args accepted (and ignored) for API compatibility with
        BasicDQNAgent.  Callers should call push() before train_step().

        Returns NaN metrics until warmup_steps transitions are in the buffer.
        """
        _nan = {"loss": float("nan"), "q_value": float("nan"),
                "target": float("nan"), "td_error": float("nan")}
        if len(self.buffer) < max(self.batch_size, self.warmup_steps):
            return _nan

        # --- sample ---
        if self.use_per:
            batch, per_idx, _ = self.buffer.sample(self.batch_size)
        else:
            batch   = self.buffer.sample(self.batch_size)
            per_idx = None

        states      = self._t(np.stack([t.state      for t in batch]))
        next_states = self._t(np.stack([t.next_state for t in batch]))
        actions     = torch.tensor([t.action for t in batch],
                                   dtype=torch.long,    device=self.device)
        rewards     = torch.tensor([t.reward for t in batch],
                                   dtype=torch.float32, device=self.device)
        dones       = torch.tensor([t.done   for t in batch],
                                   dtype=torch.float32, device=self.device)

        # --- Q(s, a) from online network ---
        q_sa = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # --- TD target from frozen target network (§2.3) ---
        with torch.no_grad():
            next_q = self.target(next_states).max(dim=1).values
            td_tgt = rewards + self.gamma * (1.0 - dones) * next_q

        loss = F.mse_loss(q_sa, td_tgt)

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        td_errors = (td_tgt - q_sa).detach().cpu().numpy()
        if self.use_per and per_idx is not None:
            self.buffer.update_priorities(per_idx, td_errors)

        # --- hard-copy target every N gradient steps ---
        self.grad_steps += 1
        if self.grad_steps % self.target_update_freq == 0:
            self._hard_update_target()

        return {
            "loss":     float(loss.detach().item()),
            "q_value":  float(q_sa.detach().mean().item()),
            "target":   float(td_tgt.detach().mean().item()),
            "td_error": float(np.abs(td_errors).mean()),
        }


    # Checkpoint
    def save(self, path: Path, *, cfg=None, extras: dict | None = None) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict = {
            "model_state_dict": self.online.state_dict(),
            "grad_steps":       self.grad_steps,
            "env_steps":        self.env_steps,
        }
        if cfg is not None:
            try:
                payload["config"] = asdict(cfg)
            except TypeError:
                payload["config"] = str(cfg)
        if extras:
            payload.update(extras)
        torch.save(payload, path)
        return path

    @classmethod
    def from_checkpoint(cls, path: Path,
                        state_shape: Tuple[int, int, int],
                        *, gamma: float = 0.99,
                        learning_rate: float = 1e-4,
                        device: torch.device = DEVICE) -> "DQNAgent":
        agent = cls(state_shape, gamma=gamma, learning_rate=learning_rate,
                    device=device)
        try:
            ckpt = torch.load(path, map_location=device, weights_only=False)
        except TypeError:
            ckpt = torch.load(path, map_location=device)
        agent.online.load_state_dict(ckpt.get("model_state_dict", ckpt))
        agent._hard_update_target()
        agent.online.eval()
        return agent