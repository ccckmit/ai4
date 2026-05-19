"""
world/core.py
~~~~~~~~~~~~~~
Core abstractions for world: the base Env class and StepResult dataclass.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Tuple, TypeVar

import numpy as np

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


@dataclass
class StepResult(Generic[ObsType]):
    """Structured return value from Env.step().

    Attributes
    ----------
    observation : ObsType
    reward : float
    terminated : bool  -- terminal state reached
    truncated : bool   -- time limit / artificial cutoff
    info : dict        -- auxiliary diagnostics
    """

    observation: ObsType
    reward: float
    terminated: bool
    truncated: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def __iter__(self):
        yield self.observation
        yield self.reward
        yield self.terminated
        yield self.truncated
        yield self.info

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Env(abc.ABC, Generic[ObsType, ActType]):
    """The main world environment interface.

    Subclasses must implement:
        reset(), step(), observation_space, action_space

    Optionally override:
        render(), close(), seed()
    """

    metadata: Dict[str, Any] = {}
    reward_range: Tuple[float, float] = (-float("inf"), float("inf"))

    # ------------------------------------------------------------------ #
    #  Abstract interface                                                  #
    # ------------------------------------------------------------------ #

    @property
    @abc.abstractmethod
    def observation_space(self):
        """Space describing valid observations."""

    @property
    @abc.abstractmethod
    def action_space(self):
        """Space describing valid actions."""

    @abc.abstractmethod
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[ObsType, Dict[str, Any]]:
        """Reset to initial state; return (obs, info)."""

    @abc.abstractmethod
    def step(self, action: ActType) -> StepResult:
        """Advance one step; return StepResult."""

    # ------------------------------------------------------------------ #
    #  Optional overrides                                                  #
    # ------------------------------------------------------------------ #

    def render(self) -> Optional[Any]:
        return None

    def close(self) -> None:
        pass

    def seed(self, seed: Optional[int] = None) -> List[int]:
        self._np_random = np.random.default_rng(seed)
        return [seed or 0]

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    def _init_rng(self, seed: Optional[int]) -> np.random.Generator:
        self._np_random = np.random.default_rng(seed)
        return self._np_random

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
