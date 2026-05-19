"""
world/envs/cartpole.py
~~~~~~~~~~~~~~~~~~~~~~~
CartPole-v1: balance a pole on a moving cart.

State vector (4 floats)
-----------------------
0  x          cart position          [-4.8, 4.8]
1  x_dot      cart velocity          (-inf, inf)
2  theta      pole angle (radians)   [-0.418, 0.418]
3  theta_dot  pole angular velocity  (-inf, inf)

Actions
-------
0 : push cart LEFT
1 : push cart RIGHT

Episode ends (terminated) when
--------------------------------
- |x|     > 2.4
- |theta| > 12° (≈0.2094 rad)

Episode truncated when
-----------------------
- steps >= max_steps (default 500)

Reward
------
+1 for every step the pole remains upright.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core import Env, StepResult
from ..spaces import Box, Discrete


class CartPoleEnv(Env[np.ndarray, int]):
    """Classic cart-pole balancing problem.

    Parameters
    ----------
    max_steps : int
        Episode truncation limit (default 500, same as gym CartPole-v1).
    render_mode : str or None
        ``"ansi"`` for text rendering.
    """

    metadata = {"render_modes": ["ansi"]}
    reward_range = (0.0, 1.0)

    # Physics constants (match OpenAI Gym exactly)
    GRAVITY = 9.8
    MASS_CART = 1.0
    MASS_POLE = 0.1
    TOTAL_MASS = MASS_CART + MASS_POLE
    HALF_POLE_LENGTH = 0.5   # half the pole's length
    POLE_MASS_LENGTH = MASS_POLE * HALF_POLE_LENGTH
    FORCE_MAG = 10.0
    TAU = 0.02  # seconds per step

    # Termination thresholds
    X_THRESHOLD = 2.4
    THETA_THRESHOLD_RAD = 12 * math.pi / 180  # 12 degrees

    def __init__(
        self,
        max_steps: int = 500,
        render_mode: Optional[str] = None,
    ):
        self._max_steps = max_steps
        self._render_mode = render_mode

        high_obs = np.array(
            [
                self.X_THRESHOLD * 2,
                np.finfo(np.float32).max,
                self.THETA_THRESHOLD_RAD * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )
        self._observation_space = Box(-high_obs, high_obs, dtype=np.float32)
        self._action_space = Discrete(2)

        self._state: Optional[np.ndarray] = None
        self._steps = 0
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
        # Initialise state uniformly at random in [-0.05, 0.05]
        self._state = self._np_random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        self._steps = 0
        return self._state.copy(), {}

    def step(self, action: int) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be 0 (left) or 1 (right).")

        self._steps += 1
        x, x_dot, theta, theta_dot = self._state

        force = self.FORCE_MAG if action == 1 else -self.FORCE_MAG
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        # Equations of motion (Euler integration)
        temp = (force + self.POLE_MASS_LENGTH * theta_dot**2 * sin_theta) / self.TOTAL_MASS
        theta_acc = (self.GRAVITY * sin_theta - cos_theta * temp) / (
            self.HALF_POLE_LENGTH * (4.0 / 3.0 - self.MASS_POLE * cos_theta**2 / self.TOTAL_MASS)
        )
        x_acc = temp - self.POLE_MASS_LENGTH * theta_acc * cos_theta / self.TOTAL_MASS

        x        += self.TAU * x_dot
        x_dot    += self.TAU * x_acc
        theta    += self.TAU * theta_dot
        theta_dot += self.TAU * theta_acc

        self._state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)

        terminated = bool(
            x < -self.X_THRESHOLD
            or x > self.X_THRESHOLD
            or theta < -self.THETA_THRESHOLD_RAD
            or theta > self.THETA_THRESHOLD_RAD
        )
        truncated = (not terminated) and self._steps >= self._max_steps
        reward = 0.0 if terminated else 1.0

        info = {
            "steps": self._steps,
            "x": float(x),
            "theta_deg": float(math.degrees(theta)),
        }
        return StepResult(self._state.copy(), reward, terminated, truncated, info)

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "ansi") -> str:
        if self._state is None:
            return ""

        x, _, theta, _ = self._state
        width = 60
        cart_col = int((x + self.X_THRESHOLD) / (2 * self.X_THRESHOLD) * (width - 1))
        cart_col = max(0, min(width - 1, cart_col))

        # Pole tip offset
        pole_tip_offset = int(math.sin(theta) * 10)
        pole_col = max(0, min(width - 1, cart_col + pole_tip_offset))

        # Draw track
        track = ["-"] * width
        track[cart_col] = "█"

        # Draw pole above cart
        pole_row = [" "] * width
        # line from cart to pole tip
        for col in range(min(cart_col, pole_col), max(cart_col, pole_col) + 1):
            pole_row[col] = "/"  if theta > 0 else "\\"
        pole_row[pole_col] = "O"

        theta_deg = math.degrees(theta)
        status = (
            f"  x={x:+.3f}  θ={theta_deg:+.1f}°  steps={self._steps}"
        )

        out = (
            "┌" + "─" * width + "┐\n"
            "│" + "".join(pole_row) + "│\n"
            "│" + "".join(track) + "│\n"
            "└" + "─" * width + "┘\n"
            + status
        )
        print(out)
        return out
