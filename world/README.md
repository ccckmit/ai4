# world

A lightweight reinforcement learning environment framework inspired by OpenAI Gym,
written in pure Python + NumPy — no heavy dependencies.

## Installation

```bash
pip install numpy
# or install as a package:
pip install -e .
```

## Quick Start

```python
import world

# Create an environment by ID
env = world.make("CartPole-v1")
obs, info = env.reset(seed=42)

for step in range(200):
    action = env.action_space.sample()   # random agent
    result = env.step(action)
    obs, reward, terminated, truncated, info = result   # tuple unpack
    if result.done:
        break

env.close()
```

## Environments

| ID | Description |
|----|-------------|
| `FrozenLake-v0` | 4×4 grid, deterministic |
| `FrozenLake-v1` | 4×4 grid, slippery (stochastic) |
| `FrozenLake8x8-v1` | 8×8 grid, slippery |
| `CartPole-v1` | Balance a pole on a cart |

## Core Concepts

### `Env` — base class
All environments inherit from `world.Env` and implement:
- `reset(seed=None)` → `(obs, info)`
- `step(action)` → `StepResult`
- `observation_space` property
- `action_space` property

### `StepResult`
```python
result.observation   # next state
result.reward        # float
result.terminated    # bool — terminal state
result.truncated     # bool — time limit hit
result.info          # dict
result.done          # terminated or truncated
obs, rew, term, trunc, info = result  # tuple unpack
```

### Spaces
```python
world.Discrete(4)                      # {0, 1, 2, 3}
world.Box(-1.0, 1.0, shape=(4,))       # continuous box
space.sample()                          # random sample
space.contains(x)                       # membership check
```

### Wrappers
```python
from world.wrappers import TimeLimitWrapper, RecordEpisodeWrapper

env = TimeLimitWrapper(env, max_steps=100)
env = RecordEpisodeWrapper(env)
# ... run episodes ...
print(env.summary())
```

### Registry
```python
world.register("MyEnv-v0", MyEnvClass, param=value)
env = world.make("MyEnv-v0")
```

## Examples

```bash
python examples/frozen_lake_example.py   # Q-Learning on FrozenLake
python examples/cartpole_example.py      # PD controller on CartPole
python tests/test_world.py              # run all tests
```

## Custom Environment

```python
import world
from world import Env, StepResult, Discrete
import numpy as np

class MyEnv(Env):
    @property
    def observation_space(self): return Discrete(10)

    @property
    def action_space(self): return Discrete(2)

    def reset(self, *, seed=None, options=None):
        self._init_rng(seed)
        self._state = 0
        return self._state, {}

    def step(self, action):
        self._state = (self._state + action) % 10
        reward = 1.0 if self._state == 9 else 0.0
        terminated = self._state == 9
        return StepResult(self._state, reward, terminated, False)

world.register("MyEnv-v0", MyEnv)
env = world.make("MyEnv-v0")
```
