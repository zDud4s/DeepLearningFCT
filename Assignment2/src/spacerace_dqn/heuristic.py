"""Hand-coded heuristic helpers

The functions here decode the SpaceRace grid (from `info["semantic_obs"]` if
it's available, otherwise from the raw RGB frame) and pick an action
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

GRID_H = 18
GRID_W = 13


def extract_semantic_position(semantic_obs: np.ndarray
                              ) -> Tuple[Optional[int], Optional[int], np.ndarray]:
    ship_channel = semantic_obs[:, :, 0]
    debris_channel = semantic_obs[:, :, 1] > 0.5
    ship_positions = np.argwhere(ship_channel > 0.5)
    if len(ship_positions) == 0:
        return None, None, debris_channel
    row, col = ship_positions[0]
    return int(row), int(col), debris_channel


def extract_rgb_grid(obs: np.ndarray, grid_h: int = GRID_H, grid_w: int = GRID_W
                     ) -> Tuple[Optional[int], Optional[int], np.ndarray]:
    """Recover ship and debris cells from the RGB observation"""
    arr = np.asarray(obs)

    # Sometimes a semantic obs gets passed in by mistake during debugging
    if arr.ndim == 3 and arr.shape[:2] == (grid_h, grid_w) and arr.max() <= 1.5:
        return extract_semantic_position(arr.astype(np.float32))

    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB observation, got shape {arr.shape}")

    if arr.dtype != np.uint8:
        if float(np.max(arr)) <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = arr.shape[:2]
    cell_h = max(1, h // grid_h)
    cell_w = max(1, w // grid_w)
    cropped = arr[: grid_h * cell_h, : grid_w * cell_w, :3]
    grid = cropped.reshape(grid_h, cell_h, grid_w, cell_w, 3)
    mean_rgb = grid.mean(axis=(1, 3))

    r = mean_rgb[:, :, 0]
    g = mean_rgb[:, :, 1]
    b = mean_rgb[:, :, 2]

    ship = (g > 150) & (b > 170) & (r < 160)
    debris = (r > 170) & (g > 120) & (b < 140)

    ship_positions = np.argwhere(ship)
    if len(ship_positions) == 0:
        return None, None, debris

    row, col = sorted(ship_positions.tolist(), key=lambda rc: (rc[0], rc[1]))[-1]
    return int(row), int(col), debris


def danger_score(debris: np.ndarray, row: int, col: int,
                 max_speed: int = 3, side_margin: int = 1) -> int:
    """Count debris cells in a horizontal window in front of (row, col)"""
    if row < 0 or row >= debris.shape[0]:
        return 0
    start_col = max(0, col - max_speed - side_margin)
    end_col = min(debris.shape[1], col + side_margin + 1)
    return int(debris[row, start_col:end_col].sum())


def action_from_grid(ship_row: Optional[int], ship_col: Optional[int],
                     debris: np.ndarray) -> int:
    """Go up unless the cell above is strictly more dangerous than below"""
    if ship_row is None or ship_col is None:
        return 0
    if ship_row <= 0:
        return 0
    up_row = max(0, ship_row - 1)
    down_row = min(debris.shape[0] - 1, ship_row + 1)
    up_danger = danger_score(debris, up_row, ship_col)
    down_danger = danger_score(debris, down_row, ship_col)
    if up_danger == 0 or up_danger <= down_danger:
        return 0
    return 1
