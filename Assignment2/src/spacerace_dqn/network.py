"""Q-network architectures"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class SmallQNetwork(nn.Module):
    """Small CNN sized for the (C, 54, 39) SpaceRace RGB observation

    Three strided conv layers (16, 32, 32) bring the feature map down to
    (32, 7, 5), then a 128-unit MLP outputs Q-values for the two actions —
    roughly 158k params with 3 input channels (no frame stacking)
    """

    def __init__(self, input_channels: int = 3, n_actions: int = 2,
                 height: int = 54, width: int = 39):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
