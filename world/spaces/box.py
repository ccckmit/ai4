"""world/spaces/box.py"""
from __future__ import annotations
from typing import Optional, Tuple, Union
import numpy as np


class Box:
    """A (possibly unbounded) box in R^n.

    Parameters
    ----------
    low, high : array-like or scalar
        Lower/upper bounds.
    shape : tuple, optional
        Shape of the space; inferred from low/high if not given.
    dtype : np.dtype
        Numeric type of samples.
    """

    def __init__(
        self,
        low: Union[float, np.ndarray],
        high: Union[float, np.ndarray],
        shape: Optional[Tuple[int, ...]] = None,
        dtype=np.float32,
        seed: Optional[int] = None,
    ):
        self.dtype = np.dtype(dtype)

        if shape is None:
            low_arr = np.asarray(low, dtype=self.dtype)
            high_arr = np.asarray(high, dtype=self.dtype)
            shape = np.broadcast_shapes(low_arr.shape, high_arr.shape) or (1,)
        else:
            shape = tuple(shape)

        self.shape = shape
        self.low = np.full(shape, low, dtype=self.dtype)
        self.high = np.full(shape, high, dtype=self.dtype)
        self._rng = np.random.default_rng(seed)

    def sample(self) -> np.ndarray:
        high = self.high.copy().astype(np.float64)
        high[high == np.inf] = 3.4e38
        low = self.low.copy().astype(np.float64)
        low[low == -np.inf] = -3.4e38
        return (self._rng.random(self.shape) * (high - low) + low).astype(self.dtype)

    def contains(self, x) -> bool:
        x = np.asarray(x, dtype=self.dtype)
        return x.shape == self.shape and bool(np.all(x >= self.low) and np.all(x <= self.high))

    def seed(self, seed: Optional[int] = None):
        self._rng = np.random.default_rng(seed)

    def __repr__(self):
        return f"Box({self.low}, {self.high}, shape={self.shape}, dtype={self.dtype})"

    def __eq__(self, other):
        return (
            isinstance(other, Box)
            and self.shape == other.shape
            and np.allclose(self.low, other.low)
            and np.allclose(self.high, other.high)
        )
