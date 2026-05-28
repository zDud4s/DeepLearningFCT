"""Open a pygame window and watch a policy play in real time"""
from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Config
from .env import make_env, reset_env
from .heuristic import extract_rgb_grid
from .improved_heuristics import HeuristicV18FullTreeS3
from .policies import Policy


class LivePlayer:
    """Roll a policy on the pygame window — needs `pygame` installed"""

    def __init__(self, cfg: Config, *, fps: int = 10):
        self.cfg = cfg
        self.fps = max(1, int(fps))

    def play_episode(self, policy: Policy, *, difficulty: int = 0,
                     seed: int = 2026) -> Dict[str, float]:
        env = make_env(self.cfg, difficulty=difficulty,
                       include_semantic_info=policy.needs_semantic_info,
                       render_mode="human")
        env.metadata["render_fps"] = self.fps
        try:
            obs, info = reset_env(env, seed=seed)
            policy.reset()
            done = False
            steps = 0
            ep_return = 0.0
            while not done:
                action = policy.select_action(obs, info, env.action_space)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_return += float(reward)
                steps += 1
                done = bool(terminated or truncated)
            return {
                "score": env.score,
                "collisions": env.collisions,
                "return": ep_return,
                "steps": steps,
            }
        finally:
            env.close()

    def play_many(self, policy: Policy, *, difficulty: int = 0,
                  seed: int = 2026, episodes: int = 1) -> list[Dict[str, float]]:
        results: list[Dict[str, float]] = []
        for ep in range(episodes):
            result = self.play_episode(policy, difficulty=difficulty,
                                       seed=seed + ep)
            results.append(result)
        return results


# ============================================================
# h18 visualizer — env on the left, debug panel on the right
# ============================================================

class _InstrumentedH18(HeuristicV18FullTreeS3):
    """h18 that captures its planning state per tick into ``last_debug``."""
    name = "h18_debug"

    def __init__(self):
        super().__init__()
        self.last_debug: Optional[dict] = None

    def select_action(self, obs, info, action_space) -> int:
        ship_row, ship_col, debris = extract_rgb_grid(obs)
        self._update_row_speeds(debris)
        if ship_row is None or ship_col is None or ship_row <= 0:
            self.last_debug = {
                "ship": (ship_row, ship_col), "debris": debris,
                "row_speed": (self.row_speed.copy() if self.row_speed is not None else None),
                "sequences": [], "winner": (0,) * self.k, "winner_score": 0.0,
                "trajectory": [], "margin": 0.0, "forced": True,
            }
            return 0
        sequences: List[Tuple[Tuple[int, ...], float, Dict[str, int]]] = []
        for code in range(1 << self.k):
            seq = tuple((code >> i) & 1 for i in range(self.k))
            score, det = self._evaluate_with_details(debris, ship_row, ship_col, seq)
            sequences.append((seq, score, det))
        sequences.sort(key=lambda x: x[1], reverse=True)
        best_seq, best_score, _ = sequences[0]
        margin = best_score - sequences[1][1] if len(sequences) > 1 else 0.0
        trajectory = self._simulate_trajectory(debris, ship_row, ship_col, best_seq)
        self.last_debug = {
            "ship": (ship_row, ship_col), "debris": debris,
            "row_speed": self.row_speed.copy(),
            "sequences": sequences, "winner": best_seq, "winner_score": best_score,
            "trajectory": trajectory, "margin": margin, "forced": False,
        }
        return best_seq[0]

    def _evaluate_with_details(self, debris, ship_row, ship_col, actions):
        d = debris.copy()
        row, col = ship_row, ship_col
        crossings = collisions = down_count = respawn_left = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                continue
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(d.shape[0] - 1, row + 1)
                down_count += 1
            d = self._step_debris(d)
            if d[row, col]:
                collisions += 1
                row = d.shape[0] - 1
                respawn_left = 15
                continue
            if row == 0:
                crossings += 1
                row = d.shape[0] - 1
        score = crossings * 1.0 - collisions * 0.25 - down_count * 0.005
        return score, {"cross": crossings, "col": collisions, "down": down_count}

    def _simulate_trajectory(self, debris, ship_row, ship_col, actions):
        d = debris.copy()
        row, col = ship_row, ship_col
        trace: List[Tuple[int, int, str]] = []
        respawn_left = 0
        for action in actions:
            if respawn_left > 0:
                respawn_left -= 1
                trace.append((row, col, "wait"))
                continue
            if action == 0:
                row = max(0, row - 1)
            else:
                row = min(d.shape[0] - 1, row + 1)
            d = self._step_debris(d)
            if d[row, col]:
                trace.append((row, col, "X"))
                row = d.shape[0] - 1
                respawn_left = 15
                continue
            if row == 0:
                trace.append((row, col, "OK"))
                row = d.shape[0] - 1
            else:
                trace.append((row, col, "."))
        return trace


class HeuristicDebugPlayer:
    """Live pygame window: env on the left, h18 planning panel on the right.

    Iterates through (difficulty, seed) pairs sequentially in the same window.
    Optionally compares h18's pick against a DQN agent loaded from
    ``submissions/<name>/agent.py`` so disagreements are visible per tick.

    Keys:
      SPACE  pause / resume
      ENTER  skip to next episode
      ESC    quit
    """

    PANEL_WIDTH = 420
    VIEW_SCALE  = 2

    def __init__(self, cfg: Config, *, fps: int = 5,
                 dqn_submission: Optional[str] = "task2_epsilon"):
        self.cfg = cfg
        self.fps = max(1, int(fps))
        self.dqn_submission = dqn_submission
        self.dqn = self._try_load_dqn(dqn_submission) if dqn_submission else None

    def _try_load_dqn(self, name: str):
        root = Path(__file__).resolve().parents[2]
        agent_py = root / "submissions" / name / "agent.py"
        if not agent_py.exists():
            print(f"[HeuristicDebugPlayer] DQN agent.py not found at {agent_py}; "
                  "skipping DQN comparison")
            return None
        try:
            spec = importlib.util.spec_from_file_location(f"dqn_dbg_{name}", agent_py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod.Agent()
        except Exception as e:
            print(f"[HeuristicDebugPlayer] failed to load DQN '{name}': {e}")
            return None

    # --------------------------------------------------------------
    # public entry
    # --------------------------------------------------------------

    def play_many(self, plan: Sequence[Tuple[int, int]]) -> None:
        """plan = [(difficulty, seed), ...]"""
        import pygame
        if not plan:
            return

        heur = _InstrumentedH18()

        probe = make_env(self.cfg, difficulty=plan[0][0],
                         include_semantic_info=False, render_mode="rgb_array")
        reset_env(probe, seed=plan[0][1])
        sample = probe.render()
        probe.close()
        if sample is None or not isinstance(sample, np.ndarray):
            raise RuntimeError("env.render() returned no rgb_array")
        rgb_h, rgb_w = sample.shape[:2]
        view_w, view_h = rgb_w * self.VIEW_SCALE, rgb_h * self.VIEW_SCALE
        win_w = view_w + self.PANEL_WIDTH
        win_h = max(view_h, 720)

        pygame.init()
        pygame.font.init()
        screen = pygame.display.set_mode((win_w, win_h))
        pygame.display.set_caption("h18 visualizer")
        clock = pygame.time.Clock()
        font  = pygame.font.SysFont("consolas", 14) or pygame.font.Font(None, 16)
        bold  = pygame.font.SysFont("consolas", 16, bold=True) or pygame.font.Font(None, 18)

        try:
            for ep_idx, (difficulty, seed) in enumerate(plan):
                quit_now = self._play_one(screen, clock, font, bold, heur,
                                          difficulty, seed, ep_idx, len(plan),
                                          view_w, view_h)
                if quit_now:
                    return
                if ep_idx < len(plan) - 1:
                    if not self._between_episodes(screen, font, bold, ep_idx, len(plan)):
                        return
        finally:
            pygame.quit()

    # --------------------------------------------------------------
    # per-episode loop
    # --------------------------------------------------------------

    def _play_one(self, screen, clock, font, bold, heur,
                  difficulty, seed, ep_idx, total, view_w, view_h) -> bool:
        import pygame
        env = make_env(self.cfg, difficulty=difficulty,
                       include_semantic_info=False, render_mode="rgb_array")
        try:
            obs, info = reset_env(env, seed=seed)
            heur.reset()
            if self.dqn is not None and hasattr(self.dqn, "frames"):
                self.dqn.frames.clear()
            paused = False
            step = 0
            disagreements = 0
            heur_act = None
            dqn_act = None
            done = False
            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        return True
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            return True
                        if event.key == pygame.K_SPACE:
                            paused = not paused
                        if event.key == pygame.K_RETURN:
                            return False
                if paused:
                    self._draw(screen, font, bold, env, heur, dqn_act, heur_act,
                               step, ep_idx, total, difficulty, seed,
                               view_w, view_h, paused=True,
                               disagreements=disagreements)
                    pygame.display.flip()
                    clock.tick(self.fps)
                    continue
                heur_act = heur.select_action(obs, info, env.action_space)
                dqn_act  = self._dqn_action(obs)
                if dqn_act is not None and dqn_act != heur_act:
                    disagreements += 1
                obs, _, terminated, truncated, info = env.step(int(heur_act))
                step += 1
                done = bool(terminated or truncated)
                self._draw(screen, font, bold, env, heur, dqn_act, heur_act,
                           step, ep_idx, total, difficulty, seed,
                           view_w, view_h, paused=False,
                           disagreements=disagreements)
                pygame.display.flip()
                clock.tick(self.fps)
            return False
        finally:
            env.close()

    def _dqn_action(self, obs) -> Optional[int]:
        if self.dqn is None:
            return None
        try:
            return int(self.dqn.select_action(obs))
        except Exception:
            return None

    # --------------------------------------------------------------
    # rendering
    # --------------------------------------------------------------

    def _draw(self, screen, font, bold, env, heur, dqn_act, heur_act,
              step, ep_idx, total, difficulty, seed, view_w, view_h,
              *, paused: bool, disagreements: int):
        import pygame
        screen.fill((20, 20, 24))
        rgb = env.render()
        if isinstance(rgb, np.ndarray):
            surf = pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
            if self.VIEW_SCALE != 1:
                surf = pygame.transform.scale(surf, (view_w, view_h))
            screen.blit(surf, (0, 0))
        self._draw_panel(screen, font, bold, env, heur, dqn_act, heur_act,
                         step, ep_idx, total, difficulty, seed, view_w,
                         paused=paused, disagreements=disagreements)

    def _draw_panel(self, screen, font, bold, env, heur, dqn_act, heur_act,
                    step, ep_idx, total, difficulty, seed, view_w,
                    *, paused: bool, disagreements: int):
        x0 = view_w + 14
        y  = 12
        col_text = (230, 230, 230)
        col_warn = (255, 180,  80)
        col_ok   = (140, 220, 140)
        col_bad  = (240, 110, 110)
        col_dim  = (150, 150, 160)

        def line(txt, color=col_text, f=font, dy=18):
            nonlocal y
            screen.blit(f.render(txt, True, color), (x0, y))
            y += dy

        line(f"episode {ep_idx + 1}/{total}    diff={difficulty}    seed={seed}",
             f=bold, dy=22)
        line(f"step={step}   score={getattr(env, 'score', '-')}   "
             f"col={getattr(env, 'collisions', '-')}", color=col_dim, dy=20)
        if paused:
            line("PAUSED  (press SPACE)", color=col_warn, f=bold, dy=22)
        y += 4

        dbg = heur.last_debug or {}
        ship = dbg.get("ship", (None, None))
        line(f"ship   row={ship[0]}   col={ship[1]}")

        def lbl(a):
            return "UP  " if a == 0 else ("DOWN" if a == 1 else "-   ")

        line(f"h18    {lbl(heur_act)}    score={dbg.get('winner_score', 0.0):+.3f}    "
             f"margin={dbg.get('margin', 0.0):+.3f}")
        if dqn_act is not None:
            agree = dqn_act == heur_act
            line(f"dqn    {lbl(dqn_act)}    "
                 f"{'agree' if agree else 'DIFFER'}    "
                 f"diffs={disagreements}",
                 color=col_ok if agree else col_bad)
        else:
            line(f"dqn    --     (sub={self.dqn_submission or 'none'})",
                 color=col_dim)
        if dbg.get("forced"):
            line("(forced UP: ship at top or undetected)", color=col_warn)
        y += 6

        rs = dbg.get("row_speed")
        if rs is not None:
            line("row_speed  (row 0 = top):", f=bold)
            chunk = " ".join(f"{int(v)}" for v in rs)
            line(chunk, color=col_text)
            y += 4

        seqs = dbg.get("sequences", [])
        if seqs:
            line("top-6 sequences (k=4):", f=bold)
            for seq, score, det in seqs[:6]:
                arrows = "".join("U" if a == 0 else "D" for a in seq)
                line(f"  {arrows}   {score:+.3f}   "
                     f"cross={det['cross']} col={det['col']} d={det['down']}",
                     dy=16)
            y += 4

        traj = dbg.get("trajectory", [])
        if traj:
            line("winner trajectory (next 4):", f=bold)
            for i, (r, c, ev) in enumerate(traj):
                color = (col_ok if ev == "OK"
                         else col_bad if ev == "X"
                         else col_dim if ev == "wait" else col_text)
                line(f"  t+{i + 1}: row={r:2d} col={c:2d}  [{ev}]",
                     color=color, dy=16)

        y += 10
        line("SPACE pause   ENTER skip   ESC quit", color=col_dim)

    # --------------------------------------------------------------
    # transition between episodes
    # --------------------------------------------------------------

    def _between_episodes(self, screen, font, bold, ep_idx, total) -> bool:
        import pygame
        w, h = screen.get_size()
        overlay = pygame.Surface((w, 60))
        overlay.set_alpha(220)
        overlay.fill((10, 10, 20))
        screen.blit(overlay, (0, h - 60))
        screen.blit(bold.render(
            f"episode {ep_idx + 1}/{total} done — ENTER for next, ESC to quit",
            True, (230, 230, 230)), (12, h - 50))
        screen.blit(font.render(
            "(window stays on the last frame)",
            True, (160, 160, 170)), (12, h - 28))
        pygame.display.flip()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_RETURN:
                        return True
                    if event.key == pygame.K_ESCAPE:
                        return False
            pygame.time.wait(20)
