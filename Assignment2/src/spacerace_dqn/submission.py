"""Build the Codabench submission package: agent.py, checkpoint, zip"""
from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional, Tuple

from .config import Config

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]
SUBMISSIONS_DIR = ASSIGNMENT_DIR / "submissions"
STARTING_KIT_DIR = ASSIGNMENT_DIR / "starting_kit"


AGENT_PY_TEMPLATE = '''from pathlib import Path
from collections import deque

import numpy as np
import torch
import torch.nn as nn


FRAME_STACK = {frame_stack}
INPUT_CHANNELS = {input_channels}
HEIGHT = {height}
WIDTH = {width}
CHECKPOINT_NAME = "{checkpoint_name}"


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
'''


class SubmissionPackager:
    """Builds the submission folder and zip, and mirrors files to the starting kit"""

    def __init__(self, cfg: Config, *, submissions_root: Path = SUBMISSIONS_DIR,
                 starting_kit: Path = STARTING_KIT_DIR):
        self.cfg = cfg
        self.submissions_root = Path(submissions_root)
        self.starting_kit = Path(starting_kit)
        self.submission_dir = self.submissions_root / cfg.submission_name
        self.submission_dir.mkdir(parents=True, exist_ok=True)

    def package(self, agent, *, state_shape: Tuple[int, int, int],
                history: list, eval_history: list, baselines: list,
                final_eval: dict, train_minutes: float,
                mirror_starting_kit: bool = True) -> dict:
        checkpoint_path = self.save_checkpoint(agent, history, eval_history, baselines)
        agent_py_path = self.export_agent_py(state_shape)
        zip_path = self.build_zip(files=["agent.py", self.cfg.checkpoint_name])
        if mirror_starting_kit:
            self.mirror_to_starting_kit()
        summary = {
            "config": asdict(self.cfg),
            "train_minutes": float(train_minutes),
            "final_eval": final_eval,
            "baselines": baselines,
            "files": {
                "checkpoint": str(checkpoint_path),
                "agent_py": str(agent_py_path),
                "zip": str(zip_path),
            },
        }
        (self.submission_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def save_checkpoint(self, agent, history: list, eval_history: list,
                        baselines: list) -> Path:
        checkpoint_path = self.submission_dir / self.cfg.checkpoint_name
        agent.save(checkpoint_path, cfg=self.cfg, extras={
            "history": history,
            "eval_history": eval_history,
            "baselines": baselines,
        })
        (self.submission_dir / "config.json").write_text(
            json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")
        (self.submission_dir / "history.json").write_text(
            json.dumps({"train": history, "eval": eval_history,
                        "baselines": baselines}, indent=2),
            encoding="utf-8")
        return checkpoint_path

    def export_agent_py(self, state_shape: Tuple[int, int, int]) -> Path:
        channels, height, width = state_shape
        path = self.submission_dir / "agent.py"
        path.write_text(AGENT_PY_TEMPLATE.format(
            frame_stack=self.cfg.frame_stack,
            input_channels=channels,
            height=height,
            width=width,
            checkpoint_name=self.cfg.checkpoint_name,
        ), encoding="utf-8")
        return path

    def build_zip(self, *, files: Iterable[str],
                  zip_name: Optional[str] = None) -> Path:
        zip_name = zip_name or f"submission_{self.cfg.submission_name}.zip"
        zip_path = self.submission_dir / zip_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for fname in files:
                zf.write(self.submission_dir / fname, arcname=fname)
        return zip_path

    def mirror_to_starting_kit(self) -> None:
        self.starting_kit.mkdir(parents=True, exist_ok=True)
        for name in ("agent.py", self.cfg.checkpoint_name):
            src = self.submission_dir / name
            if src.exists():
                (self.starting_kit / name).write_bytes(src.read_bytes())
