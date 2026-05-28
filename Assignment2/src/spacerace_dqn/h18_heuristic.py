"""h18_full_tree_s3 — standalone, self-contained version.

This module is a clean copy of the ``HeuristicV18FullTreeS3`` planner from
``improved_heuristics.py``, isolated so it can be (a) imported on its own
from notebooks and submission code, and (b) inspected without dragging in
the rest of the heuristic-experimentation suite.

Algorithm:

1. Decode the 18x13 grid from the (54, 39, 3) RGB observation by averaging
   3x3 pixel blocks and thresholding on colour:
   - cyan (R<160 & G>150 & B>170) -> ship cell
   - tan/yellow (R>170 & G>120 & B<140) -> debris cells
2. Estimate per-row debris speed by correlating the current frame with the
   previous one. Tie-break to the higher shift (safer assumption). The
   ``max_speed=3`` setting matters on difficulty 2/3: random layouts can
   contain speed-3 lanes that ``max_speed=2`` (h16) mis-identifies.
3. At every tick, enumerate all 2**4 = 16 future action sequences over the
   next 4 ticks. Simulate forward with hand-coded physics:
   - apply ship action (up=0, down=1),
   - shift each debris row by its estimated speed,
   - on collision: 15-tick respawn delay,
   - on reaching row 0: +1 crossing, respawn at row 17.
   Score = crossings - 0.25 * collisions - 0.005 * down_moves.
4. Return the first action of the highest-scoring sequence.

This is the policy used inside ``task2_h18_planner`` and inside the
``task2_ensemble`` submission's planner branch. RGB-only -- never touches
``info["semantic_obs"]`` -- so it is Codabench-legal and complies with the
"no semantic info during training" rule.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

GRID_H = 18
GRID_W = 13
MAX_SPEED = 3
PLAN_HORIZON = 4
RESPAWN_DELAY = 15


def extract_rgb_grid(obs: np.ndarray) -> Tuple[Optional[int], Optional[int], np.ndarray]:
    """Decode a (54, 39, 3) RGB observation into (ship_row, ship_col, debris_mask).

    ``debris_mask`` is a (18, 13) boolean array, ``ship_row`` and ``ship_col``
    are None when the ship is mid-respawn (no cyan cell visible).
    """
    arr = np.asarray(obs)
    if arr.dtype != np.uint8:
        if float(np.max(arr)) <= 1.5:
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        else:
            arr = np.clip(arr, 0, 255).astype(np.uint8)

    h, w = arr.shape[:2]
    cell_h = max(1, h // GRID_H)
    cell_w = max(1, w // GRID_W)
    cropped = arr[: GRID_H * cell_h, : GRID_W * cell_w, :3]
    grid = cropped.reshape(GRID_H, cell_h, GRID_W, cell_w, 3)
    mean_rgb = grid.mean(axis=(1, 3))

    r = mean_rgb[:, :, 0]
    g = mean_rgb[:, :, 1]
    b = mean_rgb[:, :, 2]
    ship = (g > 150) & (b > 170) & (r < 160)
    debris = (r > 170) & (g > 120) & (b < 140)

    ship_positions = np.argwhere(ship)
    if len(ship_positions) == 0:
        return (None, None, debris)
    # Lowest+rightmost cell — handles ship-cell rendering edge cases.
    row, col = sorted(ship_positions.tolist(), key=lambda rc: (rc[0], rc[1]))[-1]
    return (int(row), int(col), debris)


class H18Heuristic:
    """Full-tree planner, k=4, max_speed=3.

    Stateful: needs ``self.prev_debris`` to estimate per-row debris speeds.
    Call ``reset()`` between episodes so the speed estimator starts fresh.
    """

    name = "h18_full_tree_s3"

    def __init__(self, k: int = PLAN_HORIZON, max_speed: int = MAX_SPEED) -> None:
        self.k = int(k)
        self.max_speed = int(max_speed)
        self.prev_debris: Optional[np.ndarray] = None
        self.row_speed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_debris = None
        self.row_speed = None


    # Speed estimator
    def _update_row_speeds(self, debris: np.ndarray) -> None:
        """Estimate each row's debris shift via correlation with previous frame.

        Tie-breaks to the higher speed (safer to over-estimate than under-).
        Speeds are clamped to ``[1, max_speed]`` because the env never produces
        stationary debris.
        """
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return

        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, self.max_speed
            for shift in range(self.max_speed, -1, -1):
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()

    # ------------------------------------------------------------------
    # Physics simulation
    # ------------------------------------------------------------------

    def _step_debris(self, debris: np.ndarray) -> np.ndarray:
        out = debris.copy()
        for r in range(out.shape[0]):
            s = int(self.row_speed[r]) if self.row_speed is not None else self.max_speed
            out[r] = np.roll(out[r], s)
            if s > 0:
                out[r, :s] = 0
        return out

    def _evaluate_sequence(self, debris: np.ndarray, ship_row: int, ship_col: int,
                           actions: Tuple[int, ...]) -> float:
        debris_now = debris.copy()
        row, col = ship_row, ship_col
        crossings = 0
        collisions = 0
        respawn_left = 0
        down_count = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                continue
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(debris_now.shape[0] - 1, row + 1)
                down_count += 1
            debris_now = self._step_debris(debris_now)
            if debris_now[row, col]:
                collisions += 1
                row = debris_now.shape[0] - 1
                respawn_left = RESPAWN_DELAY
                continue
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
        return crossings * 1.0 - collisions * 0.25 - down_count * 0.005


    def select_action(self, obs: np.ndarray) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        best_score = -1e9
        best_first = 0
        for code in range(1 << self.k):
            seq = tuple((code >> i) & 1 for i in range(self.k))
            score = self._evaluate_sequence(debris, ship_row, ship_col, seq)
            if score > best_score:
                best_score = score
                best_first = seq[0]
        return int(best_first)


__all__ = ["H18Heuristic", "extract_rgb_grid", "GRID_H", "GRID_W",
           "MAX_SPEED", "PLAN_HORIZON", "RESPAWN_DELAY"]
