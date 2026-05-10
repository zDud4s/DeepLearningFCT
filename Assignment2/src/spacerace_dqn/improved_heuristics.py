"""Heuristic variants explored for Task 1

Each variant subclasses `Policy` and has a unique `name`, so eval results
don't collide — they live in one file because they share the same RGB grid
extractor and it makes diffing rounds easier

Naming: `h<N>_<short_label>` — h1 is the baseline, higher numbers add features
The dumb baselines (always-up, random) stay in `policies.py`
"""
from __future__ import annotations

from collections import deque
from typing import Optional, Tuple

import numpy as np

from .heuristic import (
    GRID_H,
    GRID_W,
    extract_rgb_grid,
)
from .policies import Policy

# Collision physics in SpaceRace.step():
#   1) ship moves up/down
#   2) debris moves right by its row speed
#   3) collision check


DEBRIS_LENGTH = 2


def _will_collide_window(col: int, max_speed: int) -> Tuple[int, int]:
    """Closed interval of source columns that hit `col` after one debris step"""
    lo = col - max_speed - (DEBRIS_LENGTH - 1)
    hi = col - 1  # speed is always >= 1, so x = col can't hit
    return lo, hi


def _danger_in_row(debris: np.ndarray, row: int, col: int,
                   *, max_speed: int) -> int:
    """How many debris cells on `row` will land on `col` after one tick"""
    if row < 0 or row >= debris.shape[0]:
        return 0
    lo, hi = _will_collide_window(col, max_speed)
    lo = max(0, lo)
    hi = min(debris.shape[1] - 1, hi)
    if hi < lo:
        return 0
    return int(debris[row, lo:hi + 1].sum())


def _present_in_cell(debris: np.ndarray, row: int, col: int) -> bool:
    """True if this cell is already occupied (so sitting still would hit)"""
    if row < 0 or row >= debris.shape[0] or col < 0 or col >= debris.shape[1]:
        return False
    return bool(debris[row, col])


# h1 — baseline, kept here so the comparison table uses the exact same code path

class HeuristicV1Baseline(Policy):
    """Same algorithm as `RGBHeuristicPolicy`, re-exported for the comparison run"""
    name = "h1_baseline"

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)
        # Original wide window from the first version (max_speed=3, side_margin=1)
        up_danger = _legacy_danger(debris, up_row, ship_col)
        down_danger = _legacy_danger(debris, down_row, ship_col)
        if up_danger == 0 or up_danger <= down_danger:
            return 0
        return 1


def _legacy_danger(debris, row, col, max_speed=3, side_margin=1):
    if row < 0 or row >= debris.shape[0]:
        return 0
    start_col = max(0, col - max_speed - side_margin)
    end_col = min(debris.shape[1], col + side_margin + 1)
    return int(debris[row, start_col:end_col].sum())


# h2 — tight collision window using the actual physics

class HeuristicV2TightWindow(Policy):
    """Use the real collision window derived from the env step ordering

    Only counts debris cells that would actually land on the ship's column
    after the next debris move
    """
    name = "h2_tight_window"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)
        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=self.max_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=self.max_speed)
        if up_danger == 0 or up_danger <= down_danger:
            return 0
        return 1


# h3 — aggressive: only wait when the next-tick collision is actually unavoidable

class HeuristicV3Aggressive(Policy):
    """Treat waiting as expensive — wait only when up is dangerous and down is safer

    Also refuses to wait twice in a row: that's 4 ticks for at most one extra
    dodge, almost never worth it
    """
    name = "h3_aggressive"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)
        self.last_action: Optional[int] = None

    def reset(self) -> None:
        self.last_action = None

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            self.last_action = 0
            return 0
        if ship_row <= 0:
            self.last_action = 0
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)
        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=self.max_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=self.max_speed)

        action = 0
        if up_danger >= 1 and down_danger == 0:
            if self.last_action != 1:
                action = 1
        self.last_action = action
        return action


# h4 — look two cells ahead so we don't dodge into another trap

class HeuristicV4Lookahead(Policy):
    """Like h3, but if going up now lands me below another dangerous cell,
    fall back to going down (so we don't immediately need another wait)
    """
    name = "h4_lookahead"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)
        self.last_action: Optional[int] = None

    def reset(self) -> None:
        self.last_action = None

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            self.last_action = 0
            return 0
        if ship_row <= 0:
            self.last_action = 0
            return 0
        up_row = max(0, ship_row - 1)
        upper_row = max(0, ship_row - 2)
        down_row = min(debris.shape[0] - 1, ship_row + 1)

        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=self.max_speed)
        upper_danger = _danger_in_row(debris, upper_row, ship_col, max_speed=self.max_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=self.max_speed)

        action = 0
        if up_danger >= 1 and down_danger == 0:
            if self.last_action != 1:
                action = 1
        elif up_danger == 0 and upper_danger >= 2:
            # Going up would land me at up_row, but the row above that is busy
            # Wait once if it would help (and only when down is safe)
            if down_danger == 0 and self.last_action != 1:
                action = 1
        self.last_action = action
        return action


# h5 — keep last frame, estimate per-row debris speed by correlating the shifts

class HeuristicV5Stateful(Policy):
    """Track the previous debris grid and guess per-row speed from the shift
    that best matches the current frame — use that for a row-specific window
    """
    name = "h5_stateful"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)
        self.prev_debris: Optional[np.ndarray] = None
        self.row_speed: Optional[np.ndarray] = None
        self.last_action: Optional[int] = None

    def reset(self) -> None:
        self.prev_debris = None
        self.row_speed = None
        self.last_action = None

    def _update_row_speeds(self, debris: np.ndarray) -> None:
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, self.max_speed
            for shift in range(0, self.max_speed + 1):
                # Shift the previous row right by `shift` and overlap with the current one
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            self.last_action = 0
            return 0
        if ship_row <= 0:
            self.last_action = 0
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)

        up_speed = int(self.row_speed[up_row]) if self.row_speed is not None else self.max_speed
        down_speed = int(self.row_speed[down_row]) if self.row_speed is not None else self.max_speed

        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=up_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=down_speed)

        action = 0
        if up_danger >= 1 and down_danger == 0:
            if self.last_action != 1:
                action = 1
        self.last_action = action
        return action


# h6 — full forward simulation, k-step rollout

class HeuristicV6KStepRollout(Policy):
    """Simulate `k` future ticks for each candidate action, pick the best score

    Stateful (like h5): estimates per-row debris speed from the previous frame
    and uses that to roll debris forward deterministically
    """
    name = "h6_kstep"

    def __init__(self, max_speed: int = 2, k: int = 3):
        self.max_speed = int(max_speed)
        self.k = int(k)
        self.prev_debris: Optional[np.ndarray] = None
        self.row_speed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_debris = None
        self.row_speed = None

    def _update_row_speeds(self, debris: np.ndarray) -> None:
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, self.max_speed
            for shift in range(0, self.max_speed + 1):
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()

    def _roll_debris(self, debris: np.ndarray, ticks: int) -> np.ndarray:
        rolled = debris.copy()
        for _ in range(ticks):
            for r in range(rolled.shape[0]):
                s = int(self.row_speed[r]) if self.row_speed is not None else self.max_speed
                rolled[r] = np.roll(rolled[r], s)
                # We don't know what wraps in from the right edge; assume empty
                if s > 0:
                    rolled[r, :s] = 0
        return rolled

    def _simulate(self, debris: np.ndarray, ship_row: int, ship_col: int,
                  first_action: int, depth: int) -> Tuple[int, int]:
        """Greedy roll-out: do `first_action` once, then always-up
        Returns (crossings_gained, collisions_seen)
        """
        debris_now = debris.copy()
        row, col = ship_row, ship_col
        crossings = 0
        collisions = 0
        action = first_action
        for t in range(depth):
            # 1) ship moves
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(debris_now.shape[0] - 1, row + 1)
            # 2) debris moves
            for r in range(debris_now.shape[0]):
                s = int(self.row_speed[r]) if self.row_speed is not None else self.max_speed
                debris_now[r] = np.roll(debris_now[r], s)
                if s > 0:
                    debris_now[r, :s] = 0
            # 3) collision check
            if debris_now[row, col]:
                collisions += 1
                row = debris_now.shape[0] - 1  # respawn
                # Skipping respawn-delay simulation here, just penalising the hit
                action = 0
                continue
            # 4) crossing
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
            action = 0
        return crossings, collisions

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0

        best_score = -1e9
        best_action = 0
        for a in (0, 1):
            crossings, collisions = self._simulate(debris, ship_row, ship_col, a, depth=self.k)
            score = crossings * 1.0 - collisions * 0.25 - (0.01 if a == 1 else -0.02)
            if score > best_score:
                best_score, best_action = score, a
        return best_action


# Adding new variants here updates ALL_VARIANTS automatically (via the appends below)

ALL_VARIANTS: list[Policy] = [
    HeuristicV1Baseline(),
    HeuristicV2TightWindow(max_speed=2),
    HeuristicV3Aggressive(max_speed=2),
    HeuristicV4Lookahead(max_speed=2),
    HeuristicV5Stateful(max_speed=2),
    HeuristicV6KStepRollout(max_speed=2, k=3),
]


# Round 2 — tighter, smarter waits

class HeuristicV7CertainOnly(Policy):
    """Wait only when up is certainly dangerous (>=1) AND down is fully safe
    Skip waiting if down is also dangerous — no point dodging into another trap
    """
    name = "h7_certain_only"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)
        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=self.max_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=self.max_speed)
        if up_danger > 0 and down_danger == 0 and up_danger > down_danger:
            return 1
        return 0


class HeuristicV8Stay(Policy):
    """Trick: when waiting would help, alternate down/up so the net position
    barely moves but the timing shifts by one tick
    """
    name = "h8_stay"

    def __init__(self, max_speed: int = 2):
        self.max_speed = int(max_speed)
        self.last_action: Optional[int] = None

    def reset(self) -> None:
        self.last_action = None

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if ship_row is None or ship_col is None:
            self.last_action = 0
            return 0
        if ship_row <= 0:
            self.last_action = 0
            return 0
        up_row = max(0, ship_row - 1)
        down_row = min(debris.shape[0] - 1, ship_row + 1)
        up_danger = _danger_in_row(debris, up_row, ship_col, max_speed=self.max_speed)
        down_danger = _danger_in_row(debris, down_row, ship_col, max_speed=self.max_speed)
        action = 0
        if up_danger > 0 and down_danger == 0:
            # Don't sit at the same row twice in a row
            action = 1 if self.last_action != 1 else 0
        self.last_action = action
        return action


class HeuristicV9KStepDeep(HeuristicV6KStepRollout):
    """Same as h6 but with a deeper rollout"""
    name = "h9_kstep_deep"

    def __init__(self, max_speed: int = 2, k: int = 5):
        super().__init__(max_speed=max_speed, k=k)


class HeuristicV10KStepHonestRespawn(Policy):
    """k-step rollout that handles the 15-tick respawn after a collision

    h6 was undercounting collision cost, which biased it towards accepting too
    many crashes
    """
    name = "h10_kstep_honest"

    def __init__(self, max_speed: int = 2, k: int = 6):
        self.max_speed = int(max_speed)
        self.k = int(k)
        self.prev_debris: Optional[np.ndarray] = None
        self.row_speed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_debris = None
        self.row_speed = None

    def _update_row_speeds(self, debris: np.ndarray) -> None:
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            # When uncertain, bias towards max_speed; guessing too low causes crashes
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, self.max_speed
            for shift in range(self.max_speed, -1, -1):  # tie-break towards higher speed
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            # Speed 0 never happens in this env, so clamp to >= 1
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()

    def _simulate(self, debris: np.ndarray, ship_row: int, ship_col: int,
                  first_action: int, depth: int) -> Tuple[int, int, int]:
        """Returns (crossings, collisions, ticks_used)"""
        debris_now = debris.copy()
        row, col = ship_row, ship_col
        crossings = 0
        collisions = 0
        ticks = 0
        respawn_left = 0
        action = first_action
        for _ in range(depth):
            if respawn_left > 0:
                respawn_left -= 1
                ticks += 1
                continue
            # 1) ship moves
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(debris_now.shape[0] - 1, row + 1)
            # 2) debris moves
            for r in range(debris_now.shape[0]):
                s = int(self.row_speed[r]) if self.row_speed is not None else self.max_speed
                debris_now[r] = np.roll(debris_now[r], s)
                if s > 0:
                    debris_now[r, :s] = 0
            # 3) collision check
            if debris_now[row, col]:
                collisions += 1
                row = debris_now.shape[0] - 1
                respawn_left = 15  # actual respawn delay
                action = 0
                ticks += 1
                continue
            # 4) crossing
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
            action = 0
            ticks += 1
        return crossings, collisions, ticks

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        best_score = -1e9
        best_action = 0
        for a in (0, 1):
            crossings, collisions, _ = self._simulate(
                debris, ship_row, ship_col, a, depth=self.k,
            )
            # Mirror the env reward: +1 crossing, -0.25 collision, -0.01 down
            # Only the first action contributes to the down/up bonus here
            score = crossings * 1.0 - collisions * 0.25 + (
                0.02 if a == 0 else -0.01
            )
            if score > best_score:
                best_score, best_action = score, a
        return best_action


ROUND2_VARIANTS: list[Policy] = [
    HeuristicV7CertainOnly(max_speed=2),
    HeuristicV8Stay(max_speed=2),
    HeuristicV9KStepDeep(max_speed=2, k=5),
    HeuristicV10KStepHonestRespawn(max_speed=2, k=6),
]


# Round 3 — push the rollout horizon further

class HeuristicV11KStep10(HeuristicV6KStepRollout):
    """h6 with k=10 (a 1-second horizon)"""
    name = "h11_kstep_10"

    def __init__(self, max_speed: int = 2):
        super().__init__(max_speed=max_speed, k=10)


class HeuristicV12KStep15(HeuristicV6KStepRollout):
    """k=15 fits a full ascent inside the rollout horizon"""
    name = "h12_kstep_15"

    def __init__(self, max_speed: int = 2):
        super().__init__(max_speed=max_speed, k=15)


class HeuristicV13KStepCheapCrash(HeuristicV10KStepHonestRespawn):
    """Honest respawn but discount the crossing reward more gently, so a
    collision is worth it when it saves a lot of ticks
    """
    name = "h13_kstep_cheap_crash"

    def __init__(self, max_speed: int = 2, k: int = 10):
        super().__init__(max_speed=max_speed, k=k)

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        best_score = -1e9
        best_action = 0
        for a in (0, 1):
            crossings, collisions, ticks = self._simulate(
                debris, ship_row, ship_col, a, depth=self.k,
            )
            # Score progress per tick, which naturally penalises sitting still
            progress = crossings - 0.10 * collisions
            score = progress / max(1, ticks) + (0.001 if a == 0 else -0.001)
            if score > best_score:
                best_score, best_action = score, a
        return best_action


class HeuristicV14KStep20(HeuristicV6KStepRollout):
    """k=20 covers a full ascent plus the first respawn"""
    name = "h14_kstep_20"

    def __init__(self, max_speed: int = 2):
        super().__init__(max_speed=max_speed, k=20)


ROUND3_VARIANTS: list[Policy] = [
    HeuristicV11KStep10(max_speed=2),
    HeuristicV12KStep15(max_speed=2),
    HeuristicV13KStepCheapCrash(max_speed=2, k=10),
    HeuristicV14KStep20(max_speed=2),
]


# Round 4 — drop the always-up assumption and search the full action tree

class HeuristicV15FullTree(Policy):
    """Branch and bound over every action sequence of length k

    Picks the first action of the best sequence — uses honest collision
    physics with a 15-tick respawn delay
    """
    name = "h15_full_tree"

    def __init__(self, max_speed: int = 2, k: int = 6):
        self.max_speed = int(max_speed)
        self.k = int(k)
        self.prev_debris: Optional[np.ndarray] = None
        self.row_speed: Optional[np.ndarray] = None

    def reset(self) -> None:
        self.prev_debris = None
        self.row_speed = None

    def _update_row_speeds(self, debris: np.ndarray) -> None:
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, self.max_speed
            # Tie-break towards higher speed (safer assumption)
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
        ticks = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                ticks += 1
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
                respawn_left = 15
                ticks += 1
                continue
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
            ticks += 1
        # Crossings minus collisions, with a small penalty per down move
        return crossings * 1.0 - collisions * 0.25 - down_count * 0.005

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None:
            return 0
        if ship_row <= 0:
            return 0
        # Enumerate every 2**k sequence
        best_score = -1e9
        best_first = 0
        for code in range(1 << self.k):
            seq = tuple((code >> i) & 1 for i in range(self.k))
            score = self._evaluate_sequence(debris, ship_row, ship_col, seq)
            if score > best_score:
                best_score = score
                best_first = seq[0]
        return best_first


class HeuristicV16FullTreeShallow(HeuristicV15FullTree):
    """k=4 — only 16 sequences, much faster"""
    name = "h16_full_tree_k4"

    def __init__(self, max_speed: int = 2):
        super().__init__(max_speed=max_speed, k=4)


class HeuristicV17FullTreeMid(HeuristicV15FullTree):
    """k=8 — 256 sequences"""
    name = "h17_full_tree_k8"

    def __init__(self, max_speed: int = 2):
        super().__init__(max_speed=max_speed, k=8)


ROUND4_VARIANTS: list[Policy] = [
    HeuristicV15FullTree(max_speed=2, k=6),
    HeuristicV16FullTreeShallow(max_speed=2),
    HeuristicV17FullTreeMid(max_speed=2),
]


# Round 5 — fix max_speed for difficulty 1, where leaderboard hits 32

class HeuristicV18FullTreeS3(HeuristicV15FullTree):
    """h16 with max_speed=3 so the estimator can detect speed-3 debris (diff 1)"""
    name = "h18_full_tree_s3"

    def __init__(self):
        super().__init__(max_speed=3, k=4)


class HeuristicV19FullTreeS3K6(HeuristicV15FullTree):
    """max_speed=3, deeper rollout (k=6 = 64 sequences)"""
    name = "h19_full_tree_s3_k6"

    def __init__(self):
        super().__init__(max_speed=3, k=6)


class HeuristicV20AdaptiveSpeed(HeuristicV15FullTree):
    """Estimator allowed up to speed 3, but ties go to the lower shift —
    less cautious in diff 0, where the actual max is 2
    """
    name = "h20_adaptive_speed"

    def __init__(self):
        super().__init__(max_speed=3, k=4)

    def _update_row_speeds(self, debris):
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, 1  # default to slow
            # Iterating low -> high makes ties resolve to the lower shift
            for shift in range(0, self.max_speed + 1):
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()


class HeuristicV21AutoMaxSpeed(HeuristicV15FullTree):
    """Self-tuning: starts at max_speed=3, then clamps to the actual observed
    maximum row-speed after a few frames so it doesn't stay over-cautious in diff 0
    """
    name = "h21_auto_max_speed"

    def __init__(self):
        super().__init__(max_speed=3, k=4)
        self._observed_speeds: list[int] = []

    def reset(self) -> None:
        super().reset()
        self._observed_speeds = []
        self.max_speed = 3

    def _update_row_speeds(self, debris):
        # Bootstrap with max_speed=3
        if self.prev_debris is None or self.prev_debris.shape != debris.shape:
            self.row_speed = np.full(debris.shape[0], self.max_speed, dtype=int)
            self.prev_debris = debris.copy()
            return
        speeds = np.full(debris.shape[0], self.max_speed, dtype=int)
        for r in range(debris.shape[0]):
            best_score, best_shift = -1, 1
            for shift in range(0, self.max_speed + 1):
                shifted = np.roll(self.prev_debris[r], shift)
                if shift > 0:
                    shifted[:shift] = 0
                score = int(np.logical_and(shifted, debris[r]).sum())
                if score > best_score:
                    best_score, best_shift = score, shift
            speeds[r] = max(1, best_shift)
        self.row_speed = speeds
        self.prev_debris = debris.copy()
        # After 50 frames, clamp our policy max_speed to the observed maximum
        if len(self._observed_speeds) < 50:
            self._observed_speeds.append(int(speeds.max()))
            if len(self._observed_speeds) == 50:
                self.max_speed = max(2, max(self._observed_speeds))


ROUND5_VARIANTS: list[Policy] = [
    HeuristicV18FullTreeS3(),
    HeuristicV19FullTreeS3K6(),
    HeuristicV20AdaptiveSpeed(),
    HeuristicV21AutoMaxSpeed(),
]


# Round 6 — close the gap to 32 in diff 1

class HeuristicV22NoDownPenalty(HeuristicV15FullTree):
    """h18 but without the down-move penalty (don't bias against waits)"""
    name = "h22_no_down_penalty"

    def __init__(self):
        super().__init__(max_speed=3, k=4)

    def _evaluate_sequence(self, debris, ship_row, ship_col, actions):
        debris_now = debris.copy()
        row, col = ship_row, ship_col
        crossings = 0
        collisions = 0
        respawn_left = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                continue
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(debris_now.shape[0] - 1, row + 1)
            debris_now = self._step_debris(debris_now)
            if debris_now[row, col]:
                collisions += 1
                row = debris_now.shape[0] - 1
                respawn_left = 15
                continue
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
        return crossings * 1.0 - collisions * 0.25


class HeuristicV23ProgressBonus(HeuristicV15FullTree):
    """Add a small bonus for moving up, to mirror the env reward shaping"""
    name = "h23_progress_bonus"

    def __init__(self):
        super().__init__(max_speed=3, k=4)

    def _evaluate_sequence(self, debris, ship_row, ship_col, actions):
        debris_now = debris.copy()
        row, col = ship_row, ship_col
        crossings = 0
        collisions = 0
        respawn_left = 0
        up_count = 0
        down_count = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                continue
            if action == 0:
                row = max(0, row - 1)
                up_count += 1
            else:
                row = min(debris_now.shape[0] - 1, row + 1)
                down_count += 1
            debris_now = self._step_debris(debris_now)
            if debris_now[row, col]:
                collisions += 1
                row = debris_now.shape[0] - 1
                respawn_left = 15
                continue
            if row == 0:
                crossings += 1
                row = debris_now.shape[0] - 1
        # Env reward shaping: +0.02 per up, -0.01 per down
        return crossings * 1.0 - collisions * 0.25 + up_count * 0.02 - down_count * 0.01


class HeuristicV24K8(HeuristicV15FullTree):
    """k=8 with max_speed=3 (256 sequences per call)"""
    name = "h24_k8_s3"

    def __init__(self):
        super().__init__(max_speed=3, k=8)


class HeuristicV25Resimulate(HeuristicV15FullTree):
    """Try to predict debris re-entry at the left edge

    When debris exits the right edge during the rollout, drop a copy at
    column -2 so the planner sees its eventual return — only matters for
    horizons longer than the cycle
    """
    name = "h25_resimulate"

    def __init__(self):
        super().__init__(max_speed=3, k=6)

    def _step_debris(self, debris):
        out = debris.copy()
        speeds = (self.row_speed if self.row_speed is not None
                  else np.full(out.shape[0], self.max_speed, dtype=int))
        for r in range(out.shape[0]):
            s = int(speeds[r])
            # Anything in the rightmost `s` cells is about to exit this tick
            exiting = int(out[r, max(0, out.shape[1] - s):].sum())
            out[r] = np.roll(out[r], s)
            if s > 0:
                out[r, :s] = 0
            # If something exited, re-spawn an equivalent piece near the left
            # Without phase info we just place a DEBRIS_LENGTH-wide piece at col 0
            if exiting > 0:
                out[r, 0] = 1
                if DEBRIS_LENGTH > 1:
                    out[r, 1] = 1
        return out


ROUND6_VARIANTS: list[Policy] = [
    HeuristicV22NoDownPenalty(),
    HeuristicV23ProgressBonus(),
    HeuristicV24K8(),
    HeuristicV25Resimulate(),
]


# ---------------------------------------------------------------------------
# Iteration round 7 — mirror the exact env physics for non-random difficulties
# ---------------------------------------------------------------------------

class _LaneMirror:
    """Mirrors `SpaceRaceEnv` debris physics for a given non-random difficulty

    Replicates the deterministic lane layout (rows, speeds, initial cols) and
    the right-edge reset formula — only suitable for diff 0 and diff 1 where
    `randomize_debris=False`
    """
    def __init__(self, difficulty: int, width: int = 13, height: int = 18,
                 debris_length: int = 2, density: float = 0.6):
        self.width = width
        self.height = height
        self.debris_length = debris_length
        self.spacing = max(2, debris_length + 1)
        self.wrap_width = width + debris_length
        speed_bonus = 1 if difficulty >= 1 else 0
        self.min_speed = 1 + speed_bonus
        self.max_speed = 2 + speed_bonus
        self.density = density + (0.15 if difficulty >= 3 else 0.0)

        candidate_rows = list(range(1, height - 1))
        target_count = max(1, int(round(len(candidate_rows) * self.density)))
        step = len(candidate_rows) / float(target_count)
        rows: list[int] = []
        for i in range(target_count):
            idx = int(i * step + step / 2)
            idx = min(idx, len(candidate_rows) - 1)
            row = candidate_rows[idx]
            if row not in rows:
                rows.append(row)
        if len(rows) < target_count:
            for row in candidate_rows:
                if row not in rows:
                    rows.append(row)
                if len(rows) == target_count:
                    break

        speed_span = max(1, self.max_speed - self.min_speed + 1)
        self.debris: list[dict] = []
        for lane_id, row in enumerate(rows):
            col = -debris_length - (lane_id * self.spacing)
            speed = self.min_speed + (lane_id % speed_span)
            self.debris.append({"row": row, "col": col,
                                "speed": speed, "lane_id": lane_id})

    def clone(self) -> "_LaneMirror":
        cl = object.__new__(_LaneMirror)
        cl.width = self.width
        cl.height = self.height
        cl.debris_length = self.debris_length
        cl.spacing = self.spacing
        cl.wrap_width = self.wrap_width
        cl.min_speed = self.min_speed
        cl.max_speed = self.max_speed
        cl.density = self.density
        cl.debris = [dict(d) for d in self.debris]
        return cl

    def step(self) -> None:
        speed_span = max(1, self.max_speed - self.min_speed + 1)
        for d in self.debris:
            d["col"] += d["speed"]
            if d["col"] > self.width - 1:
                # The env's _reset_debris_item recomputes lane_id from the
                # debris row (NOT the original enumerate index) — it also
                # recomputes speed using the same row-derived lane_id, so a
                # debris piece can switch speed after its first cycle, so we
                # have to replicate that exactly to stay in sync
                reset_lane = max(0, d["row"] - 1)
                d["col"] = -self.debris_length - (
                    (reset_lane * self.spacing) % self.wrap_width
                )
                d["speed"] = self.min_speed + (reset_lane % speed_span)

    def grid(self) -> np.ndarray:
        out = np.zeros((self.height, self.width), dtype=bool)
        for d in self.debris:
            for s in range(self.debris_length):
                c = d["col"] + s
                if 0 <= c < self.width:
                    out[d["row"], c] = True
        return out


class HeuristicV26MirrorEnv(Policy):
    """Plan against an internal mirror env that replicates exact debris physics

    Workflow per call:
      1. Maintain candidate mirrors for diff 0 and diff 1
      2. Step each mirror once per tick
      3. After a few frames, lock to the mirror whose grid best matches the
         observed RGB grid (diff 2/3 mirrors won't match well)
      4. If no mirror matches, fall back to h18-style planning with
         max_speed=3
      5. Plan k=4 ahead using the locked mirror's exact `step()`
    """
    name = "h26_mirror_env"

    def __init__(self, k: int = 4):
        self.k = int(k)
        self.candidates: list[_LaneMirror] = []
        self.locked: Optional[_LaneMirror] = None
        # max_speed=2 fallback empirically beats max_speed=3 on random
        # difficulties: a narrower danger window means fewer false-positive
        # waits, and the few extra collisions from missed speed-3 debris
        # are net-cheap.
        self.fallback_max_speed = 2
        self._frame_count = 0
        self._mismatch_history: list[tuple[int, int]] = []  # per-candidate cumulative mismatch

    def reset(self) -> None:
        self.candidates = [_LaneMirror(difficulty=0), _LaneMirror(difficulty=1)]
        self.locked = None
        self._frame_count = 0
        self._mismatch_history = [0 for _ in self.candidates]

    def _step_all_candidates(self) -> None:
        for c in self.candidates:
            c.step()

    def _maybe_lock(self, observed: np.ndarray) -> None:
        if self.locked is not None:
            return
        for i, c in enumerate(self.candidates):
            mismatch = int(np.logical_xor(c.grid(), observed).sum())
            self._mismatch_history[i] += mismatch
        if self._frame_count < 5:
            return
        best_i = int(np.argmin(self._mismatch_history))
        best_total = self._mismatch_history[best_i]
        # Two safeguards before locking:
        #   (1) absolute threshold: avg <= 1 mismatch / frame means we
        #       genuinely match a deterministic mirror (random difficulties
        #       never satisfy this)
        #   (2) relative threshold: best must be clearly better than others
        if best_total > 1 * (self._frame_count + 1):
            return
        others = [s for j, s in enumerate(self._mismatch_history) if j != best_i]
        if not others or best_total < min(others):
            self.locked = self.candidates[best_i]

    def _h18_fallback(self, debris, ship_row, ship_col) -> int:
        """Cheap planner without the mirror — same as h18/h22"""
        max_speed = self.fallback_max_speed
        # Rebuild a tiny inline planner that uses the *observed* debris grid
        # only (plus a wide max_speed) — same as h18/h22's evaluate_sequence
        best_score, best_first = -1e18, 0
        for code in range(1 << self.k):
            seq = tuple((code >> i) & 1 for i in range(self.k))
            grid = debris.copy()
            row, col = ship_row, ship_col
            crossings = 0
            collisions = 0
            respawn_left = 0
            for action in seq:
                if respawn_left > 0:
                    respawn_left -= 1
                    continue
                if action == 0:
                    row = max(0, row - 1)
                else:
                    row = min(grid.shape[0] - 1, row + 1)
                # No mirror: just shift each row by max_speed
                for r in range(grid.shape[0]):
                    grid[r] = np.roll(grid[r], max_speed)
                    grid[r, :max_speed] = 0
                if grid[row, col]:
                    collisions += 1
                    row = grid.shape[0] - 1
                    respawn_left = 15
                    continue
                if row == 0:
                    crossings += 1
                    row = grid.shape[0] - 1
            score = crossings * 1.0 - collisions * 0.25
            if score > best_score:
                best_score, best_first = score, seq[0]
        return best_first

    def _plan_with_mirror(self, ship_row: int, ship_col: int) -> int:
        """Branch over 2**k action sequences using the locked mirror"""
        best_score, best_first = -1e18, 0
        for code in range(1 << self.k):
            seq = tuple((code >> i) & 1 for i in range(self.k))
            mirror = self.locked.clone()
            row, col = ship_row, ship_col
            crossings = 0
            collisions = 0
            respawn_left = 0
            for action in seq:
                if respawn_left > 0:
                    respawn_left -= 1
                    continue
                if action == 0:
                    row = max(0, row - 1)
                else:
                    row = min(mirror.height - 1, row + 1)
                mirror.step()
                if mirror.grid()[row, col]:
                    collisions += 1
                    row = mirror.height - 1
                    respawn_left = 15
                    continue
                if row == 0:
                    crossings += 1
                    row = mirror.height - 1
            score = crossings * 1.0 - collisions * 0.25
            if score > best_score:
                best_score, best_first = score, seq[0]
        return best_first

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        if not self.candidates:
            self.reset()
        # IMPORTANT: at this point mirror state corresponds to *current* obs
        # (NOT next tick) — compare BEFORE stepping, plan from current state
        self._maybe_lock(debris)
        self._frame_count += 1

        action = 0
        if ship_row is None or ship_col is None:
            action = 0
        elif ship_row <= 0:
            action = 0
        elif self.locked is not None:
            action = self._plan_with_mirror(ship_row, ship_col)
        else:
            action = self._h18_fallback(debris, ship_row, ship_col)

        # Now advance the mirror by one tick so it matches the *next* obs
        self._step_all_candidates()
        return action


ROUND7_VARIANTS: list[Policy] = [
    HeuristicV26MirrorEnv(k=4),
]

ALL_VARIANTS = (ALL_VARIANTS + ROUND2_VARIANTS + ROUND3_VARIANTS
                + ROUND4_VARIANTS + ROUND5_VARIANTS + ROUND6_VARIANTS
                + ROUND7_VARIANTS)
