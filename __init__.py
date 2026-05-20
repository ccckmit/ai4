"""ai4 - DIY AI Framework for Python.

Subpackages
-----------
world : Reinforcement learning environments
nn    : Neural network framework (NumPy-based)

Examples
--------
>>> from ai4 import world, nn
>>> env = world.make("FrozenLake-v1")
>>> from ai4 import nn
>>> from nn import GPT
"""

from . import world
from . import nn

__all__ = ["world", "nn"]