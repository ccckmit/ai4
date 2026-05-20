"""world/spaces/discrete.py"""
from __future__ import annotations

from typing import Optional
import numpy as np


class Discrete:
    """A discrete space of n integers {0, 1, ..., n-1}."""

    def __init__(self, n: int, start: int = 0, seed: Optional[int] = None) -> None:
        assert n >= 1, "n must be >= 1"
        self.n: int = int(n)
        self.start: int = int(start)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> int:
        return int(self._rng.integers(self.start, self.start + self.n))

    def contains(self, x: object) -> bool:
        return isinstance(x, (int, np.integer)) and self.start <= x < self.start + self.n

    def seed(self, seed: Optional[int] = None) -> None:
        self._rng = np.random.default_rng(seed)

    @property
    def shape(self) -> tuple[()]:
        return ()

    @property
    def dtype(self) -> np.dtype:
        return np.int64

    def __repr__(self) -> str:
        return f"Discrete({self.n}, start={self.start})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Discrete) and self.n == other.n and self.start == other.start