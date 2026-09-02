"""
world/envs/pong.py
~~~~~~~~~~~~~~~~~~~
Pong: single-player paddle-ball game.

State vector (5 floats)
-----------------------
0  ball_x    ball x position          [0, 1]
1  ball_y    ball y position          [0, 1]
2  ball_vx   ball x velocity          [-0.03, 0.03]
3  ball_vy   ball y velocity          [-0.03, 0.03]
4  paddle_y  paddle center y position [0, 1]

Actions
-------
0 : move paddle UP
1 : move paddle DOWN

Episode ends (terminated) when
-------------------------------
- ball passes left wall (x < 0)

Episode truncated when
----------------------
- steps >= max_steps (default 1000)

Reward
------
+1 per step while ball is in play,
-1 when ball misses the paddle.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core import Env, StepResult
from ..spaces import Box, Discrete


class PongEnv(Env[np.ndarray, int]):
    """Single-player Pong environment.

    Parameters
    ----------
    max_steps : int
        Episode truncation limit (default 1000).
    render_mode : str or None
        ``"ansi"`` for text rendering.
    """

    metadata = {"render_modes": ["ansi"]}
    reward_range = (-1.0, 1.0)

    BALL_SPEED = 0.015
    PADDLE_SPEED = 0.04
    PADDLE_HALF = 0.1
    PADDLE_X = 0.04

    def __init__(
        self,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ):
        self._max_steps = max_steps
        self._render_mode = render_mode

        high_obs = np.array([1.0, 1.0, self.BALL_SPEED * 2, self.BALL_SPEED * 2, 1.0], dtype=np.float32)
        low_obs = np.array([0.0, 0.0, -self.BALL_SPEED * 2, -self.BALL_SPEED * 2, 0.0], dtype=np.float32)
        self._observation_space = Box(low_obs, high_obs, dtype=np.float32)
        self._action_space = Discrete(2)

        self.ball_x = 0.0
        self.ball_y = 0.0
        self.ball_vx = 0.0
        self.ball_vy = 0.0
        self.paddle_y = 0.5
        self._steps = 0
        self._hits = 0
        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------ #
    #  Space properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Discrete:
        return self._action_space

    # ------------------------------------------------------------------ #
    #  Core interface                                                      #
    # ------------------------------------------------------------------ #

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        self._init_rng(seed)

        angle = self._np_random.uniform(-1.0, 1.0) * (math.pi / 4)
        self.ball_x = 0.5
        self.ball_y = 0.5
        self.ball_vx = -self.BALL_SPEED * math.cos(angle)
        self.ball_vy = self.BALL_SPEED * math.sin(angle)
        self.paddle_y = 0.5
        self._steps = 0
        self._hits = 0

        obs = np.array([self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_y], dtype=np.float32)
        return obs, {}

    def step(self, action: int) -> StepResult:
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 (UP) or 1 (DOWN).")

        self._steps += 1

        # move paddle
        if action == 0:
            self.paddle_y = max(0.0, self.paddle_y - self.PADDLE_SPEED)
        else:
            self.paddle_y = min(1.0, self.paddle_y + self.PADDLE_SPEED)

        # move ball
        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy

        # bounce off top/bottom
        if self.ball_y <= 0.0:
            self.ball_y = 0.0
            self.ball_vy = abs(self.ball_vy)
        if self.ball_y >= 1.0:
            self.ball_y = 1.0
            self.ball_vy = -abs(self.ball_vy)

        # bounce off right wall
        if self.ball_x >= 1.0:
            self.ball_x = 1.0
            self.ball_vx = -abs(self.ball_vx)

        # paddle collision
        hit = (
            self.ball_x <= self.PADDLE_X + 0.02
            and self.ball_x >= self.PADDLE_X - 0.02
            and abs(self.ball_y - self.paddle_y) <= self.PADDLE_HALF
        )
        if hit:
            self.ball_vx = abs(self.ball_vx)
            self.ball_x = self.PADDLE_X + 0.02
            spin = self._np_random.uniform(-0.005, 0.005)
            self.ball_vy += spin
            self.ball_vy = max(-self.BALL_SPEED, min(self.BALL_SPEED, self.ball_vy))
            self._hits += 1

        # check miss
        terminated = bool(self.ball_x < 0.0)
        truncated = (not terminated) and self._steps >= self._max_steps
        reward = -1.0 if terminated else 1.0

        obs = np.array([self.ball_x, self.ball_y, self.ball_vx, self.ball_vy, self.paddle_y], dtype=np.float32)
        info = {"steps": self._steps, "hits": self._hits}
        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self, mode: Optional[str] = None) -> Optional[str]:
        if mode is None:
            mode = self._render_mode or "ansi"

        if mode == "ansi":
            W, H = 40, 16
            grid = [[" "] * W for _ in range(H)]

            for x in range(W):
                grid[0][x] = "─"
                grid[H - 1][x] = "─"
            for y in range(H):
                grid[y][0] = "│"
                grid[y][W - 1] = "│"
            grid[0][0] = "┌"
            grid[0][W - 1] = "┐"
            grid[H - 1][0] = "└"
            grid[H - 1][W - 1] = "┘"

            # paddle
            px = int(self.PADDLE_X * (W - 3)) + 1
            py_center = int(self.paddle_y * (H - 2)) + 1
            phalf = int(math.ceil(self.PADDLE_HALF * (H - 2)))
            for dy in range(phalf + 1):
                y1 = max(1, py_center - dy)
                y2 = min(H - 2, py_center + dy)
                if 0 < y1 < H - 1:
                    grid[y1][px] = "█"
                if 0 < y2 < H - 1:
                    grid[y2][px] = "█"

            # ball
            bx = int(self.ball_x * (W - 3)) + 1
            by = int(self.ball_y * (H - 2)) + 1
            if 0 < by < H - 1 and 0 < bx < W - 1:
                grid[by][bx] = "●"

            out = "\n".join("".join(row) for row in grid)
            out += f"\n  hits={self._hits}  steps={self._steps}"
            print(out)
            return out

        return None

    def close(self):
        return None
