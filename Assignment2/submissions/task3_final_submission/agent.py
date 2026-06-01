from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn


FRAME_STACK = 1
INPUT_CHANNELS = 3
HEIGHT = 54
WIDTH = 39
CHECKPOINT_NAME = "task3_exploration.pt"


class SmallQNetwork(nn.Module):
    def __init__(self, input_channels=INPUT_CHANNELS, n_actions=2, height=HEIGHT, width=WIDTH):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, height, width)
            flat_features = int(np.prod(self.features(dummy).shape[1:]))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_features, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x))


def preprocess_obs(obs):
    arr = np.asarray(obs, dtype=np.float32) / 255.0
    return np.transpose(arr, (2, 0, 1))


class Agent:
    def __init__(self):
        self.device = torch.device("cpu")
        self.model = SmallQNetwork().to(self.device)
        checkpoint_path = Path(__file__).with_name(CHECKPOINT_NAME)
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.frames = deque(maxlen=FRAME_STACK)

    def _state_from_obs(self, obs):
        frame = preprocess_obs(obs)
        if not self.frames:
            for _ in range(FRAME_STACK):
                self.frames.append(frame)
        else:
            self.frames.append(frame)
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)

    def select_action(self, obs):
        state = self._state_from_obs(obs)
        state_t = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_t)
        return int(torch.argmax(q_values, dim=1).item())
