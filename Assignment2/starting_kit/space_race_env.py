"""Gymnasium/Gym environment for a classic single-player Space Race."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
try:
    import gymnasium as gym
    from gymnasium import spaces
    from gymnasium.envs.registration import register
except ModuleNotFoundError:
    try:
        import gym
        from gym import spaces
        from gym.envs.registration import register
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "SpaceRace requires 'gymnasium' (preferred) or 'gym'. "
            "Install one with: pip install gymnasium numpy"
        ) from exc
try:
    import pygame
except ModuleNotFoundError:
    pygame = None


@dataclass(frozen=True)
class RewardConfig:
    """Reward values used by the environment."""

    reach_top: float = 1.0
    collision: float = -0.25
    move_up: float = 0.02
    move_down: float = -0.01


@dataclass
class Debris:
    """Single debris dash moving horizontally on one row."""

    row: int
    col: int
    speed: int


class SpaceRaceEnv(gym.Env):
    """
    Classic Space Race adaptation for single-agent RL.

    Rules modeled from the arcade game:
    - The player travels from bottom to top repeatedly.
    - Debris moves horizontally across rows.
    - Collisions do not end the round; the ship respawns at the bottom.
    - The episode is limited by time (default: 60 seconds).
    """

    metadata = {
        "render_modes": ["human", "rgb_array", "ansi"],
        "render_fps": 10,
    }

    def __init__(
        self,
        width: int = 13,
        height: int = 18,
        round_time_seconds: float = 60.0,
        ticks_per_second: int = 10,
        difficulty: int = 0,
        debris_density: float = 0.6,
        debris_length: int = 2,
        min_debris_speed: int = 1,
        max_debris_speed: int = 2,
        respawn_seconds: float = 1.5,
        render_mode: str | None = None,
        reward_config: RewardConfig | None = None,
        obs_mode: str = "rgb",
        include_semantic_info: bool = True,
    ) -> None:
        if width < 5:
            raise ValueError("width must be >= 5")
        if height < 6:
            raise ValueError("height must be >= 6")
        if round_time_seconds <= 0:
            raise ValueError("round_time_seconds must be > 0")
        if ticks_per_second < 1:
            raise ValueError("ticks_per_second must be >= 1")
        if difficulty not in (0, 1, 2, 3):
            raise ValueError("difficulty must be one of {0, 1, 2, 3}")
        if not 0.1 <= debris_density <= 1.0:
            raise ValueError("debris_density must be in [0.1, 1.0]")
        if debris_length < 1 or debris_length >= width:
            raise ValueError("debris_length must be in [1, width-1]")
        if min_debris_speed < 1:
            raise ValueError("min_debris_speed must be >= 1")
        if max_debris_speed < min_debris_speed:
            raise ValueError("max_debris_speed must be >= min_debris_speed")
        if respawn_seconds < 0:
            raise ValueError("respawn_seconds must be >= 0")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"Unsupported render_mode: {render_mode}")
        if obs_mode not in ("semantic", "rgb"):
            raise ValueError(f"obs_mode must be 'semantic' or 'rgb', got: {obs_mode}")

        self.width = width
        self.height = height
        self.round_time_seconds = round_time_seconds
        self.ticks_per_second = ticks_per_second
        self.round_steps = max(1, int(round(round_time_seconds * ticks_per_second)))
        self.difficulty = difficulty
        self.randomize_debris = difficulty >= 2
        self.debris_length = debris_length

        speed_bonus = 1 if difficulty >= 1 else 0
        self.min_debris_speed = min_debris_speed + speed_bonus
        self.max_debris_speed = max_debris_speed + speed_bonus
        self.debris_density = debris_density + (0.15 if difficulty >= 3 else 0.0)
        self.debris_density = min(1.0, self.debris_density)

        self.respawn_steps = int(round(respawn_seconds * ticks_per_second))
        self.render_mode = render_mode
        self.rewards = reward_config or RewardConfig()
        self.obs_mode = obs_mode
        self.include_semantic_info = include_semantic_info

        # Classic controls: move ship up or down.
        self.action_space = spaces.Discrete(2)

        # Observation space depends on obs_mode.
        if self.obs_mode == "semantic":
            # Channels: ship, debris, normalized time remaining.
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.height, self.width, 3),
                dtype=np.float32,
            )
        else:  # obs_mode == "rgb"
            # Downsampled RGB image (cell_size=3) for pixel-based learning.
            self._rgb_cell_size = 3
            rgb_height = self.height * self._rgb_cell_size
            rgb_width = self.width * self._rgb_cell_size
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(rgb_height, rgb_width, 3),
                dtype=np.uint8,
            )

        self.ship_col = self.width // 2
        self.start_row = self.height - 1

        self.ship_row = self.start_row
        self.ship_visible = True
        self.respawn_counter = 0
        self.elapsed_steps = 0
        self.score = 0
        self.collisions = 0
        self.debris: list[Debris] = []

        self.cell_size_px = 32
        self.hud_height_px = 72
        self.window_width_px = self.width * self.cell_size_px
        self.window_height_px = self.height * self.cell_size_px + self.hud_height_px
        self.window = None
        self.clock = None
        self.font = None

    def _make_debris(self) -> list[Debris]:
        candidate_rows = list(range(1, self.start_row))
        target_count = max(1, int(round(len(candidate_rows) * self.debris_density)))
        chosen_rows = self._choose_debris_rows(candidate_rows, target_count)

        items: list[Debris] = []
        for lane_id, row in enumerate(chosen_rows):
            col = self._initial_debris_col(lane_id)
            speed = self._debris_speed(lane_id)
            items.append(Debris(row=row, col=col, speed=speed))
        return items

    def _choose_debris_rows(self, candidate_rows: list[int], target_count: int) -> list[int]:
        if self.randomize_debris:
            chosen_rows = self.np_random.choice(candidate_rows, size=target_count, replace=False)
            return sorted(int(row) for row in chosen_rows)

        if target_count >= len(candidate_rows):
            return candidate_rows

        # Deterministic lane spacing for lower difficulties.
        step = len(candidate_rows) / float(target_count)
        selected: list[int] = []
        for i in range(target_count):
            idx = int(i * step + step / 2)
            idx = min(idx, len(candidate_rows) - 1)
            row = candidate_rows[idx]
            if row not in selected:
                selected.append(row)

        if len(selected) < target_count:
            for row in candidate_rows:
                if row not in selected:
                    selected.append(row)
                if len(selected) == target_count:
                    break
        return selected

    def _initial_debris_col(self, lane_id: int) -> int:
        if self.randomize_debris:
            return int(self.np_random.integers(-self.width, self.width))
        spacing = max(2, self.debris_length + 1)
        return -self.debris_length - (lane_id * spacing)

    def _debris_speed(self, lane_id: int) -> int:
        if self.randomize_debris:
            return int(self.np_random.integers(self.min_debris_speed, self.max_debris_speed + 1))
        speed_span = max(1, self.max_debris_speed - self.min_debris_speed + 1)
        return self.min_debris_speed + (lane_id % speed_span)

    def _reset_debris_item(self, item: Debris) -> None:
        lane_id = max(0, item.row - 1)
        if self.randomize_debris:
            item.col = -int(self.np_random.integers(self.width // 2, self.width + 1))
            item.speed = self._debris_speed(lane_id)
            return

        spacing = max(2, self.debris_length + 1)
        wrap_width = self.width + self.debris_length
        item.col = -self.debris_length - ((lane_id * spacing) % wrap_width)
        item.speed = self._debris_speed(lane_id)

    def _move_debris(self) -> None:
        for item in self.debris:
            item.col += item.speed
            if item.col > self.width - 1:
                self._reset_debris_item(item)

    def _ship_hit(self) -> bool:
        for item in self.debris:
            if item.row != self.ship_row:
                continue
            for segment in range(self.debris_length):
                if item.col + segment == self.ship_col:
                    return True
        return False

    def _time_remaining_ratio(self) -> float:
        remaining = max(0, self.round_steps - self.elapsed_steps)
        return remaining / float(self.round_steps)

    def _time_remaining_seconds(self) -> float:
        remaining = max(0, self.round_steps - self.elapsed_steps)
        return remaining / float(self.ticks_per_second)

    def _build_grid(self) -> list[list[str]]:
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for item in self.debris:
            for segment in range(self.debris_length):
                col = item.col + segment
                if 0 <= col < self.width:
                    grid[item.row][col] = "-"
        if self.ship_visible:
            grid[self.ship_row][self.ship_col] = "A"
        return grid

    def _get_semantic_obs(self) -> np.ndarray:
        """Return semantic observation with separate channels for ship, debris, time."""
        obs = np.zeros((self.height, self.width, 3), dtype=np.float32)
        if self.ship_visible:
            obs[self.ship_row, self.ship_col, 0] = 1.0

        for item in self.debris:
            for segment in range(self.debris_length):
                col = item.col + segment
                if 0 <= col < self.width:
                    obs[item.row, col, 1] = 1.0

        obs[:, :, 2] = self._time_remaining_ratio()
        return obs

    def _get_rgb_obs(self) -> np.ndarray:
        """Return downsampled RGB observation for pixel-based learning."""
        colors = {
            " ": np.array([5, 10, 20], dtype=np.uint8),      # Dark space background
            "-": np.array([230, 180, 70], dtype=np.uint8),   # Yellow/tan debris
            "A": np.array([90, 220, 250], dtype=np.uint8),   # Cyan ship
        }
        grid = self._build_grid()
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for r in range(self.height):
            for c in range(self.width):
                image[r, c] = colors[grid[r][c]]
        # Upsample to cell_size=3 for 54×39 output
        return np.repeat(np.repeat(image, self._rgb_cell_size, axis=0), self._rgb_cell_size, axis=1)

    def _get_obs(self) -> np.ndarray:
        """Return observation based on obs_mode."""
        if self.obs_mode == "semantic":
            return self._get_semantic_obs()
        return self._get_rgb_obs()

    def _get_info(self) -> dict[str, Any]:
        info = {
            "score": self.score,
            "completed_runs": self.score,
            "collisions": self.collisions,
            "difficulty": self.difficulty,
            "debris_density": self.debris_density,
            "elapsed_seconds": self.elapsed_steps / float(self.ticks_per_second),
            "time_remaining_seconds": self._time_remaining_seconds(),
            "ship_visible": self.ship_visible,
            "respawn_seconds_remaining": self.respawn_counter / float(self.ticks_per_second),
        }
        # Include semantic observation for heuristic development (not for NN input!)
        # Disabled during evaluation to prevent cheating.
        if self.include_semantic_info:
            info["semantic_obs"] = self._get_semantic_obs()
        return info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        del options  # Reserved for Gymnasium API compatibility.

        self.ship_row = self.start_row
        self.ship_visible = True
        self.respawn_counter = 0
        self.elapsed_steps = 0
        self.score = 0
        self.collisions = 0
        self.debris = self._make_debris()

        observation = self._get_obs()
        info = self._get_info()
        if self.render_mode == "human":
            self.render()
        return observation, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Expected one of [0, 1].")

        self.elapsed_steps += 1
        reward = 0.0

        if self.ship_visible:
            previous_row = self.ship_row
            if action == 0:
                self.ship_row = max(0, self.ship_row - 1)
            else:
                self.ship_row = min(self.start_row, self.ship_row + 1)

            if self.ship_row < previous_row:
                reward += self.rewards.move_up
            elif self.ship_row > previous_row:
                reward += self.rewards.move_down

        self._move_debris()

        if self.ship_visible and self._ship_hit():
            self.collisions += 1
            reward += self.rewards.collision
            self.ship_visible = False
            self.ship_row = self.start_row
            self.respawn_counter = self.respawn_steps
            if self.respawn_counter == 0:
                self.ship_visible = True

        if not self.ship_visible and self.respawn_counter > 0:
            self.respawn_counter -= 1
            if self.respawn_counter == 0:
                self.ship_visible = True
                self.ship_row = self.start_row

        if self.ship_visible and self.ship_row == 0:
            self.score += 1
            reward += self.rewards.reach_top
            self.ship_row = self.start_row

        terminated = False
        truncated = self.elapsed_steps >= self.round_steps

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, truncated, info

    def render(self) -> np.ndarray | str | None:
        if self.render_mode is None:
            return None

        grid = self._build_grid()

        lines = ["|" + "".join(row) + "|" for row in grid]
        lines.append("+" + "-" * self.width + "+")
        lines.append(
            f"Level: {self.difficulty}  Score: {self.score}  "
            f"Collisions: {self.collisions}  Time: {self._time_remaining_seconds():.1f}s"
        )
        text = "\n".join(lines)

        if self.render_mode == "ansi":
            return text
        if self.render_mode == "human":
            self._render_human(grid)
            return None

        colors = {
            " ": np.array([5, 10, 20], dtype=np.uint8),
            "-": np.array([230, 180, 70], dtype=np.uint8),
            "A": np.array([90, 220, 250], dtype=np.uint8),
        }
        cell_size = 12
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        for r in range(self.height):
            for c in range(self.width):
                image[r, c] = colors[grid[r][c]]
        return np.repeat(np.repeat(image, cell_size, axis=0), cell_size, axis=1)

    def _ensure_human_display(self) -> None:
        if pygame is None:
            raise ModuleNotFoundError(
                "Human display requires 'pygame'. Install it with: pip install pygame"
            )
        if self.window is not None:
            return

        pygame.init()
        pygame.display.init()
        pygame.font.init()
        self.window = pygame.display.set_mode((self.window_width_px, self.window_height_px))
        pygame.display.set_caption("SpaceRace-v0")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 28)

    def _render_human(self, grid: list[list[str]]) -> None:
        self._ensure_human_display()
        if self.window is None:
            return

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return

        space_bg = (8, 10, 18)
        cell_bg = (12, 18, 30)
        grid_line = (22, 32, 52)
        finish_line = (35, 80, 45)
        start_line = (25, 38, 80)
        debris_color = (230, 180, 70)
        ship_color = (90, 220, 250)
        hud_bg = (16, 16, 22)
        text_color = (240, 240, 240)

        self.window.fill(space_bg)
        board_height_px = self.height * self.cell_size_px

        top_rect = pygame.Rect(0, 0, self.window_width_px, self.cell_size_px)
        bottom_rect = pygame.Rect(
            0,
            (self.height - 1) * self.cell_size_px,
            self.window_width_px,
            self.cell_size_px,
        )
        pygame.draw.rect(self.window, finish_line, top_rect)
        pygame.draw.rect(self.window, start_line, bottom_rect)

        for row in range(self.height):
            for col in range(self.width):
                rect = pygame.Rect(
                    col * self.cell_size_px,
                    row * self.cell_size_px,
                    self.cell_size_px,
                    self.cell_size_px,
                )
                if row not in (0, self.height - 1):
                    pygame.draw.rect(self.window, cell_bg, rect)
                pygame.draw.rect(self.window, grid_line, rect, 1)

        for row in range(self.height):
            for col in range(self.width):
                symbol = grid[row][col]
                if symbol == "-":
                    inset = max(2, self.cell_size_px // 6)
                    rect = pygame.Rect(
                        col * self.cell_size_px + inset,
                        row * self.cell_size_px + (self.cell_size_px // 3),
                        self.cell_size_px - (2 * inset),
                        self.cell_size_px // 3,
                    )
                    pygame.draw.rect(self.window, debris_color, rect)
                elif symbol == "A":
                    center_x = col * self.cell_size_px + self.cell_size_px // 2
                    center_y = row * self.cell_size_px + self.cell_size_px // 2
                    radius = max(4, self.cell_size_px // 3)
                    pygame.draw.circle(self.window, ship_color, (center_x, center_y), radius)

        hud_rect = pygame.Rect(0, board_height_px, self.window_width_px, self.hud_height_px)
        pygame.draw.rect(self.window, hud_bg, hud_rect)

        info_text = (
            f"Level: {self.difficulty}   Score: {self.score}   Collisions: {self.collisions}   "
            f"Time: {self._time_remaining_seconds():.1f}s"
        )
        if self.font is not None:
            text_surface = self.font.render(info_text, True, text_color)
            self.window.blit(text_surface, (12, board_height_px + 22))

        pygame.display.flip()
        if self.clock is not None:
            self.clock.tick(self.metadata.get("render_fps", 10))

    def display(self) -> np.ndarray | str | None:
        """Explicit display alias for render()."""
        return self.render()

    def close(self) -> None:
        if pygame is not None:
            if self.window is not None:
                pygame.display.quit()
            pygame.quit()
        self.window = None
        self.clock = None
        self.font = None
        return None


def register_space_race_env() -> None:
    """Register the environment with Gymnasium."""
    try:
        register(
            id="SpaceRace-v0",
            entry_point="SpaceRace.space_race_env:SpaceRaceEnv",
        )
    except gym.error.Error:
        # Ignore registration errors when already registered.
        pass
