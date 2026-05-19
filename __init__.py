"""ai4py - DIY AI Framework for Python.

Subpackages
-----------
world : Reinforcement learning environments
nn    : Neural network framework (NumPy-based)

Examples
--------
>>> from ai4py import world, nn
>>> env = world.make("FrozenLake-v1")
>>> from ai4py import nn
>>> from nn import GPT
"""

from . import world
from . import nn

__all__ = ["world", "nn"]