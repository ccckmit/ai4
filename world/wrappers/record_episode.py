"""world/wrappers/record_episode.py - Records episode statistics."""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
from ..core import Env, StepResult


class RecordEpisodeWrapper(Env):
    """Collects per-episode statistics (length, cumulative reward).

    Access via ``wrapper.episode_stats`` after episodes complete.
    """

    def __init__(self, env: Env):
        self._env = env
        self.episode_stats: List[Dict[str, float]] = []
        self._ep_reward = 0.0
        self._ep_length = 0

    @property
    def observation_space(self):
        return self._env.observation_space

    @property
    def action_space(self):
        return self._env.action_space

    def reset(self, *, seed=None, options=None) -> Tuple[Any, Dict]:
        self._ep_reward = 0.0
        self._ep_length = 0
        return self._env.reset(seed=seed, options=options)

    def step(self, action) -> StepResult:
        result = self._env.step(action)
        self._ep_reward += result.reward
        self._ep_length += 1
        if result.done:
            self.episode_stats.append({
                "reward": self._ep_reward,
                "length": self._ep_length,
                "terminated": result.terminated,
                "truncated": result.truncated,
            })
        return result

    def render(self, **kw):
        return self._env.render(**kw)

    def close(self):
        self._env.close()

    def summary(self) -> Dict[str, float]:
        if not self.episode_stats:
            return {}
        rewards = [e["reward"] for e in self.episode_stats]
        lengths = [e["length"] for e in self.episode_stats]
        import statistics
        return {
            "episodes": len(rewards),
            "mean_reward": statistics.mean(rewards),
            "max_reward": max(rewards),
            "min_reward": min(rewards),
            "mean_length": statistics.mean(lengths),
        }

    def __repr__(self):
        return f"RecordEpisodeWrapper({self._env})"
