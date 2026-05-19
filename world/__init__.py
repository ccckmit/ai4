"""
world
~~~~~~
A lightweight reinforcement learning environment framework
inspired by OpenAI Gym, written in pure Python + NumPy.

Quick start
-----------
>>> import world
>>> env = world.make("FrozenLake-v1")
>>> obs, info = env.reset(seed=42)
>>> result = env.step(env.action_space.sample())
"""

from .core import Env, StepResult
from .spaces import Discrete, Box
from .envs import FrozenLakeEnv, CartPoleEnv
from .wrappers import TimeLimitWrapper, RecordEpisodeWrapper
from .utils import register, make, registry, run_random_agent

# ------------------------------------------------------------------ #
#  Built-in environment registrations                                 #
# ------------------------------------------------------------------ #
register("FrozenLake-v0",     FrozenLakeEnv, map_name="4x4", is_slippery=False)
register("FrozenLake-v1",     FrozenLakeEnv, map_name="4x4", is_slippery=True)
register("FrozenLake8x8-v1",  FrozenLakeEnv, map_name="8x8", is_slippery=True)
register("CartPole-v1",       CartPoleEnv,   max_steps=500)

__version__ = "0.1.0"
__all__ = [
    # Core
    "Env", "StepResult",
    # Spaces
    "Discrete", "Box",
    # Environments
    "FrozenLakeEnv", "CartPoleEnv",
    # Wrappers
    "TimeLimitWrapper", "RecordEpisodeWrapper",
    # Utils
    "register", "make", "registry", "run_random_agent",
]
