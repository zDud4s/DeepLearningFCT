"""Policies used by training, eval, recording and live play

Each one implements `select_action(obs, info, action_space) -> int`
"""
from __future__ import annotations

import importlib.util
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .heuristic import (
    action_from_grid,
    extract_rgb_grid,
    extract_semantic_position,
)


class Policy(ABC):
    name: str = "policy"
    needs_semantic_info: bool = False  # set True if the policy reads info["semantic_obs"]

    @abstractmethod
    def select_action(self, obs: np.ndarray, info: dict, action_space) -> int:
        ...

    def reset(self) -> None:
        """Called at the start of every episode — override if you keep state"""


class RandomPolicy(Policy):
    name = "random"

    def select_action(self, obs, info, action_space) -> int:
        return int(action_space.sample())


class AlwaysUpPolicy(Policy):
    name = "always_up"

    def select_action(self, obs, info, action_space) -> int:
        return 0


class RGBHeuristicPolicy(Policy):
    """Heuristic that decodes the grid straight from the RGB frame (Codabench-legal)"""
    name = "rgb_heuristic"

    def select_action(self, obs, info, action_space) -> int:
        return action_from_grid(*extract_rgb_grid(obs))


class SemanticHeuristicPolicy(Policy):
    """Same heuristic but reads info['semantic_obs'] — useful only as a sanity check"""
    name = "semantic_heuristic"
    needs_semantic_info = True

    def select_action(self, obs, info, action_space) -> int:
        semantic = info.get("semantic_obs")
        if semantic is None:
            return 0
        return action_from_grid(*extract_semantic_position(semantic))


class DQNInferencePolicy(Policy):
    """Loads an exported `Agent` from a submission folder and runs it

    Use this when you want to evaluate the file Codabench will actually run,
    not the in-memory training agent
    """
    name = "dqn"

    def __init__(self, agent_py_path: Path):
        self.agent_py_path = Path(agent_py_path)
        if not self.agent_py_path.exists():
            raise FileNotFoundError(f"Missing agent.py at {self.agent_py_path}")
        self._module_name = f"submitted_agent_{self.agent_py_path.parent.name}"
        if self._module_name in sys.modules:
            del sys.modules[self._module_name]
        spec = importlib.util.spec_from_file_location(self._module_name, self.agent_py_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self._agent: Any = module.Agent()

    def select_action(self, obs, info, action_space) -> int:
        return int(np.clip(int(self._agent.select_action(obs)), 0, 1))


def policy_factory(name: str, *, agent_py_path: Optional[Path] = None) -> Policy:
    """Resolve a policy by name — returns a fresh instance"""
    if name == "random":
        return RandomPolicy()
    if name == "always_up":
        return AlwaysUpPolicy()
    if name == "rgb_heuristic" or name == "heuristic":
        return RGBHeuristicPolicy()
    if name == "semantic_heuristic":
        return SemanticHeuristicPolicy()
    if name == "dqn":
        if agent_py_path is None:
            raise ValueError("policy_factory(name='dqn') requires agent_py_path")
        return DQNInferencePolicy(agent_py_path)
    raise ValueError(f"Unknown policy name: {name}")
