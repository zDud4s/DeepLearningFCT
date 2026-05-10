"""Render a policy playing one episode and save the result as GIF/MP4/PNG"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

from .config import Config
from .env import make_env, reset_env
from .policies import Policy

ASSIGNMENT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_RECORDINGS_DIR = ASSIGNMENT_DIR / "docs" / "recordings"


def _upscale(rgb: np.ndarray, scale: int) -> np.ndarray:
    return np.repeat(np.repeat(rgb, scale, axis=0), scale, axis=1)


def _annotate(rgb: np.ndarray, lines: List[str]) -> np.ndarray:
    pad_h = 22 * len(lines) + 6
    canvas = Image.new("RGB", (rgb.shape[1], rgb.shape[0] + pad_h), color=(16, 16, 22))
    canvas.paste(Image.fromarray(rgb), (0, pad_h))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    for i, line in enumerate(lines):
        draw.text((6, 4 + i * 22), line, fill=(240, 240, 240), font=font)
    return np.asarray(canvas)


class GifRecorder:
    """Records frame-by-frame and writes GIF, MP4 or a 6-frame PNG strip"""

    def __init__(self, cfg: Config, *, output_dir: Path = DEFAULT_RECORDINGS_DIR):
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def record(self, policy: Policy, *,
               difficulty: int = 0, seed: int = 2026,
               max_steps: int = 600, scale: int = 6,
               format: str = "gif", tag: str | None = None) -> Path:
        env = make_env(self.cfg, difficulty=difficulty,
                       include_semantic_info=True,
                       render_mode="rgb_array")
        obs, info = env.reset(seed=seed)
        policy.reset()

        frames: List[np.ndarray] = []
        step = 0
        done = False
        while not done and step < max_steps:
            action = policy.select_action(obs, info, env.action_space)
            rgb = env.render()
            big = _upscale(rgb, scale)
            annotated = _annotate(big, [
                f"policy={policy.name}  diff={difficulty}  seed={seed}",
                f"step={step:3d}  score={env.score}  collisions={env.collisions}  "
                f"action={'UP' if action == 0 else 'DOWN'}",
            ])
            frames.append(annotated)
            obs, _, terminated, truncated, info = env.step(int(action))
            step += 1
            done = bool(terminated or truncated)
        env.close()

        name_tag = tag or policy.name
        if format == "gif":
            path = self.output_dir / f"{name_tag}_diff{difficulty}_seed{seed}.gif"
            imageio.mimsave(path, frames, duration=0.10, loop=0)
        elif format == "mp4":
            path = self.output_dir / f"{name_tag}_diff{difficulty}_seed{seed}.mp4"
            imageio.mimsave(path, frames, fps=10)
        elif format == "png_strip":
            path = self.output_dir / f"{name_tag}_diff{difficulty}_seed{seed}.png"
            chosen = [frames[i] for i in np.linspace(0, len(frames) - 1, 6).astype(int)]
            h = chosen[0].shape[0]
            w = sum(f.shape[1] for f in chosen) + 5 * (len(chosen) - 1)
            canvas = np.full((h, w, 3), 16, dtype=np.uint8)
            x = 0
            for f in chosen:
                canvas[:, x:x + f.shape[1]] = f
                x += f.shape[1] + 5
            Image.fromarray(canvas).save(path)
        else:
            raise ValueError(f"Unknown format: {format}")
        return path
