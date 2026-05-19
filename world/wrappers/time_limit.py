"""world/wrappers/time_limit.py - Hard time-limit wrapper."""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from ..core import Env, StepResult


class TimeLimitWrapper(Env):
    """Truncates episodes after *max_steps* steps."""

    def __init__(self, env: Env, max_steps: int):
        self._env = env
        self._max_steps = max_steps
        self._elapsed = 0

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict]:
        self._elapsed = 0
        return self._env.reset(seed=seed, options=options)

    def step(self, action) -> StepResult:
        result = self._env.step(action)
        self._elapsed += 1
        if self._elapsed >= self._max_steps and not result.terminated:
            return StepResult(result.observation, result.reward, result.terminated, True, result.info)
        return result

    def render(self, **kw):
        return self._env.render(**kw)

    def close(self):
        self._env.close()

    def __repr__(self):
        return f"TimeLimitWrapper({self._env}, max_steps={self._max_steps})"
