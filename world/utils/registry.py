"""world/utils/registry.py - Environment registry (make / register)."""
from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Type
from ..core import Env

_REGISTRY: Dict[str, Dict[str, Any]] = {}


def register(
    id: str,
    entry_point: Callable[..., Env],
    **default_kwargs,
) -> None:
    """Register an environment.

    Parameters
    ----------
    id : str
        Unique environment ID, e.g. ``"FrozenLake-v1"``.
    entry_point : callable
        Class or factory function returning an :class:`Env`.
    **default_kwargs
        Default keyword arguments passed to *entry_point*.
    """
    if id in _REGISTRY:
        raise ValueError(f"Environment '{id}' is already registered.")
    _REGISTRY[id] = {"entry_point": entry_point, "kwargs": default_kwargs}


def make(id: str, **kwargs) -> Env:
    """Instantiate a registered environment.

    Parameters
    ----------
    id : str
        Registered environment ID.
    **kwargs
        Override or extend the default kwargs.
    """
    if id not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown environment '{id}'. Available: {available}")
    spec = _REGISTRY[id]
    merged = {**spec["kwargs"], **kwargs}
    return spec["entry_point"](**merged)


def registry() -> Dict[str, Dict[str, Any]]:
    """Return a copy of the full environment registry."""
    return dict(_REGISTRY)
