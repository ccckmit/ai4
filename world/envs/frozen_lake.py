"""
world/envs/frozen_lake.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
FrozenLake-v1: a grid-world where the agent navigates a frozen lake.

Map legend
----------
S : start
F : frozen (safe)
H : hole  (episode ends, reward=0)
G : goal  (episode ends, reward=1)

Actions
-------
0 : LEFT
1 : DOWN
2 : RIGHT
3 : UP
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core import Env, StepResult
from ..spaces import Discrete

# --------------------------------------------------------------------------- #
#  Built-in maps                                                               #
# --------------------------------------------------------------------------- #

MAPS: Dict[str, List[str]] = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG",
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG",
    ],
}

# Cardinal directions as (row_delta, col_delta)
_ACTIONS = {
    0: (0, -1),   # LEFT
    1: (1,  0),   # DOWN
    2: (0,  1),   # RIGHT
    3: (-1, 0),   # UP
}
_ACTION_NAMES = {0: "LEFT", 1: "DOWN", 2: "RIGHT", 3: "UP"}

# ANSI colour codes for terminal rendering
_ANSI = {
    "S": "\033[94mS\033[0m",   # blue
    "F": "\033[97mF\033[0m",   # white
    "H": "\033[91mH\033[0m",   # red
    "G": "\033[92mG\033[0m",   # green
    "A": "\033[93mA\033[0m",   # yellow (agent)
}


class FrozenLakeEnv(Env[int, int]):
    """Grid-world frozen lake.

    Parameters
    ----------
    map_name : str
        Key into the built-in ``MAPS`` dict (``"4x4"`` or ``"8x8"``).
    custom_map : list of str, optional
        Provide your own map; overrides ``map_name``.
    is_slippery : bool
        When True the agent moves in the intended direction only 1/3 of
        the time; with probability 2/3 it slides to one of the two
        perpendicular directions (standard world behaviour).
    max_steps : int, optional
        Episode length limit (truncation).  Defaults to 100 * nrow.
    """

    metadata = {"render_modes": ["ansi"]}

    def __init__(
        self,
        map_name: str = "4x4",
        custom_map: Optional[List[str]] = None,
        is_slippery: bool = True,
        max_steps: Optional[int] = None,
    ):
        desc = custom_map if custom_map is not None else MAPS[map_name]
        self._desc = [list(row) for row in desc]
        self._nrow = len(self._desc)
        self._ncol = len(self._desc[0])
        self._is_slippery = is_slippery

        # Locate start & goal
        self._start = self._find("S")
        self._holes = self._find_all("H")
        self._goal = self._find("G")

        n_states = self._nrow * self._ncol
        self._observation_space = Discrete(n_states)
        self._action_space = Discrete(4)

        self._max_steps = max_steps or (100 * self._nrow)
        self._agent_pos: Tuple[int, int] = self._start
        self._steps = 0
        self._np_random = np.random.default_rng()

    # ------------------------------------------------------------------ #
    #  Space properties                                                    #
    # ------------------------------------------------------------------ #

    @property
    def observation_space(self) -> Discrete:
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
    ) -> Tuple[int, Dict[str, Any]]:
        self._init_rng(seed)
        self._agent_pos = self._start
        self._steps = 0
        obs = self._pos_to_state(self._agent_pos)
        return obs, {"pos": self._agent_pos}

    def step(self, action: int) -> StepResult:
        if not self._action_space.contains(action):
            raise ValueError(f"Invalid action {action}. Must be in {self._action_space}.")

        self._steps += 1

        # Slippery: choose from {action, left-of-action, right-of-action}
        if self._is_slippery:
            candidates = [(action - 1) % 4, action, (action + 1) % 4]
            action = int(self._np_random.choice(candidates))

        dr, dc = _ACTIONS[action]
        r, c = self._agent_pos
        nr = min(max(r + dr, 0), self._nrow - 1)
        nc = min(max(c + dc, 0), self._ncol - 1)
        self._agent_pos = (nr, nc)

        cell = self._desc[nr][nc]
        terminated = cell in ("H", "G")
        truncated = (not terminated) and (self._steps >= self._max_steps)
        reward = 1.0 if cell == "G" else 0.0

        obs = self._pos_to_state(self._agent_pos)
        info = {
            "pos": self._agent_pos,
            "cell": cell,
            "action_taken": _ACTION_NAMES[action],
            "steps": self._steps,
        }
        return StepResult(obs, reward, terminated, truncated, info)

    # ------------------------------------------------------------------ #
    #  Rendering                                                           #
    # ------------------------------------------------------------------ #

    def render(self, mode: str = "ansi") -> str:
        ar, ac = self._agent_pos
        lines = []
        for r, row in enumerate(self._desc):
            line = ""
            for c, cell in enumerate(row):
                if (r, c) == (ar, ac):
                    line += _ANSI["A"]
                else:
                    line += _ANSI.get(cell, cell)
            lines.append(line)
        out = "\n".join(lines)
        print(out)
        return out

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _pos_to_state(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self._ncol + pos[1]

    def _find(self, char: str) -> Tuple[int, int]:
        for r, row in enumerate(self._desc):
            for c, cell in enumerate(row):
                if cell == char:
                    return (r, c)
        raise ValueError(f"Character '{char}' not found in map.")

    def _find_all(self, char: str) -> List[Tuple[int, int]]:
        return [
            (r, c)
            for r, row in enumerate(self._desc)
            for c, cell in enumerate(row)
            if cell == char
        ]
