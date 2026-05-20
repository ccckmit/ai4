"""
world/envs/bipedal_walker.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~
BipedalWalker-v3: A bipedal robot walking task.

Observation space (24 dimensions)
----------------------------------
0-3   : hull angle, hull angular velocity
4-7   : leg 1 joint angle, joint velocity (left leg)
8-11  : leg 2 joint angle, joint velocity (right leg)
12-13 : hip left joint, hip right joint
14-15 : knee left joint, knee right joint
16-23 : 8 contacts (left_leg_h, left_leg_a, right_leg_h, right_leg_a, 
                   hip1, knee1, hip2, knee2)

Action space (4 dimensions)
--------------------------
0-1   : hip joint torque (left, right)
2-3   : knee joint torque (left, right)

Episode ends (terminated) when
-----------------------------
- hull angle > 1.0 rad (tipped over)
- termination happens when the walker has trouble 
  but can still stand back up and continues

Episode truncated when
---------------------
- steps >= max_steps (default 1600)

Reward
------
- +1 for every step if the walker is not on the ground
- -0.1 for each step of torque applied
- bonus of +10 for reaching the finish line
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np

from ..core import Env, StepResult
from ..spaces import Box


class BipedalWalkerEnv(Env[np.ndarray, np.ndarray]):
    """BipedalWalker-v3: A challenging locomotion task.
    
    The walker must learn to walk without falling.
    The episode terminates when the hull touches the ground
    or the walker reaches the finish line.
    """

    metadata = {"render_modes": ["ansi", "human"]}
    reward_range = (-float("inf"), float("inf"))

    GRAVITY = 9.8
    FPS = 50
    SCALE = 30.0  # Scaling factor for rendering

    # Terrain parameters
    TERRAIN_LENGTH = 8.0
    TERRAIN_HEIGHT = 0.0
    TERRAIN_GRASS = 0.07
    TERRAIN_STARTPAD = 1.0
    TERRAIN_FINISH = 0.0

    # Walker parameters
    LEG_W = 0.2
    LEG_H = 1.0

    # Termination thresholds
    TILT_LIMIT = 1.0

    def __init__(
        self,
        max_steps: int = 1600,
        render_mode: Optional[str] = None,
    ):
        self.max_steps = max_steps
        self.render_mode = render_mode

        # Observation: 24 dims (hull: 4, leg1: 4, leg2: 4, hip: 2, knee: 2, contacts: 8)
        high_obs = np.full(24, 1.0, dtype=np.float32)
        
        self._observation_space = Box(-high_obs, high_obs, dtype=np.float32)
        # Action: 4 joint torques
        self._action_space = Box(-1.0, 1.0, shape=(4,), dtype=np.float32)

        # State: [hull_x, hull_y, hull_angle, ...]
        self._state: Optional[np.ndarray] = None
        self._steps = 0
        self._np_random = np.random.default_rng()
        self._initialized = False

        # Walker body parameters
        self._hull_pos = np.array([0.0, 3.0 * self.LEG_H / 2])
        self._hull_angle = 0.0
        self._hull_angular_vel = 0.0
        self._leg1_angle = 0.0
        self._leg1_angular_vel = 0.0
        self._leg2_angle = 0.0
        self._leg2_angular_vel = 0.0

    # ------------------------------------------------------------------ #
    #  Space properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def observation_space(self) -> Box:
        return self._observation_space

    @property
    def action_space(self) -> Box:
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
        self._steps = 0
        self._initialized = True

        # Initialize walker state
        self._hull_pos = np.array([0.0, 3.0 * self.LEG_H / 2])
        self._hull_angle = 0.0
        self._hull_angular_vel = self._np_random.uniform(-0.05, 0.05)
        self._leg1_angle = 0.0
        self._leg1_angular_vel = self._np_random.uniform(-0.05, 0.05)
        self._leg2_angle = 0.0
        self._leg2_angular_vel = self._np_random.uniform(-0.05, 0.05)

        obs = self._get_obs()
        return obs, {}

    def step(self, action: np.ndarray) -> StepResult:
        if not self._initialized:
            raise RuntimeError("Call reset() before step().")
        
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self._action_space}.")

        self._steps += 1

        # Apply action (torque)
        torque = action * 30.0  # Scale up torque

        # Simple physics update
        # Hull angle changes based on angular velocity
        self._hull_angular_vel += (torque[0] + torque[1]) * 0.01
        self._hull_angular_vel *= 0.99  # Damping
        self._hull_angle += self._hull_angular_vel * 0.1

        # Leg angles change
        self._leg1_angular_vel += torque[2] * 0.01
        self._leg1_angular_vel *= 0.99
        self._leg1_angle += self._leg1_angular_vel * 0.1

        self._leg2_angular_vel += torque[3] * 0.01
        self._leg2_angular_vel *= 0.99
        self._leg2_angle += self._leg2_angular_vel * 0.1

        # Calculate reward
        reward = self._calculate_reward()

        # Check termination
        terminated = abs(self._hull_angle) > self.TILT_LIMIT
        
        # Check truncation
        truncated = self._steps >= self.max_steps

        obs = self._get_obs()
        info = {
            "steps": self._steps,
            "hull_angle": float(self._hull_angle),
        }

        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _get_obs(self) -> np.ndarray:
        """Get observation vector - 24 dims for BipedalWalker-v3"""
        return np.array([
            self._hull_pos[0], self._hull_pos[1],  # hull position
            math.sin(self._hull_angle), self._hull_angular_vel,  # hull angle, angvel
            math.sin(self._leg1_angle), self._leg1_angular_vel,  # leg1
            math.sin(self._leg2_angle), self._leg2_angular_vel,  # leg2
            math.sin(self._leg1_angle), self._leg1_angular_vel,  # hip left
            math.sin(self._leg2_angle), self._leg2_angular_vel,  # hip right
            math.sin(self._leg1_angle), self._leg1_angular_vel,  # knee left
            math.sin(self._leg2_angle), self._leg2_angular_vel,  # knee right
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # contacts
        ], dtype=np.float32)

    def _calculate_reward(self) -> float:
        """Calculate reward based on walker state."""
        # Reward for being upright and moving forward
        # Base reward: 1.0 per step if legs are moving
        reward = 1.0

        # Penalty for using too much torque
        # (simplified - would need actual torque calculation)
        reward -= 0.01

        # Bonus for hull being above ground
        if self._hull_pos[1] > 1.0:
            reward += 0.1

        return reward

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "ansi") -> Optional[str]:
        if self._state is None:
            return None

        # Simple ANSI rendering
        width = 60
        height = 20
        
        # Convert hull position to screen coordinates
        x = int((self._hull_pos[0] / 10.0 + 0.5) * width)
        y = int((1.0 - self._hull_pos[1] / 5.0) * height)
        
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))

        # Draw
        lines = []
        for row in range(height):
            line = [" " if col != x or row != y else "█" for col in range(width)]
            lines.append("".join(line))
        
        out = "\n".join(lines)
        
        if mode == "human" or mode == "ansi":
            print(out)
        
        return out