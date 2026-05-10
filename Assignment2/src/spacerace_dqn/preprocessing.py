"""Observation preprocessing utilities"""
from __future__ import annotations

from collections import deque

import numpy as np


def preprocess_obs(obs: np.ndarray) -> np.ndarray:
    """HWC uint8 RGB -> CHW float32 in [0, 1]"""
    arr = np.asarray(obs, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class FrameStacker:
    """Keep the last N preprocessed frames concatenated along the channel axis"""

    def __init__(self, stack_size: int = 1):
        self.stack_size = int(stack_size)
        self.frames: deque[np.ndarray] = deque(maxlen=self.stack_size)

    def reset(self, obs: np.ndarray) -> np.ndarray:
        frame = preprocess_obs(obs)
        self.frames.clear()
        for _ in range(self.stack_size):
            self.frames.append(frame)
        return self.state()

    def append(self, obs: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_obs(obs))
        return self.state()

    def state(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)
