"""Convenience helpers around `SpaceRaceEnv`

The env source lives under `Assignment2/starting_kit/space_race_env.py`, so
this module makes sure that path is on `sys.path` before importing
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]
STARTING_KIT_DIR = ASSIGNMENT_DIR / "starting_kit"
if str(STARTING_KIT_DIR) not in sys.path:
    sys.path.insert(0, str(STARTING_KIT_DIR))

from starting_kit.space_race_env import SpaceRaceEnv

from .config import Config


def make_env(cfg: Config,
             *,
             difficulty: Optional[int] = None,
             include_semantic_info: Optional[bool] = None,
             render_mode: Optional[str] = None) -> SpaceRaceEnv:
    """Build a `SpaceRaceEnv` consistent with `cfg`"""
    return SpaceRaceEnv(
        difficulty=cfg.difficulty if difficulty is None else difficulty,
        obs_mode=cfg.obs_mode,
        round_time_seconds=cfg.round_time_seconds,
        ticks_per_second=cfg.ticks_per_second,
        include_semantic_info=(cfg.include_semantic_info
                               if include_semantic_info is None
                               else include_semantic_info),
        render_mode=render_mode,
    )


def reset_env(env: SpaceRaceEnv, seed: Optional[int] = None):
    """`env.reset()` that also handles the seed-less case"""
    if seed is None:
        return env.reset()
    return env.reset(seed=seed)


__all__ = ["SpaceRaceEnv", "make_env", "reset_env"]
