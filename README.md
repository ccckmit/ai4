# ai4

A DIY AI Framework for Python, consisting of two independent subpackages:

- **world/** — Lightweight reinforcement learning environment framework (pure Python + NumPy)
- **nn/** — DIY neural network framework with automatic differentiation (NumPy-based)

## Quick Start

### world (RL Environments)

```python
from ai4 import world

env = world.make("FrozenLake-v1")
obs, info = env.reset(seed=42)
result = env.step(env.action_space.sample())
```

### nn (Neural Networks)

```python
from ai4 import nn
from nn import GPT

model = GPT(vocab_size=100, block_size=16, n_layer=1, n_embd=16, n_head=4)
```

## Running Examples

```bash
# world tests
cd /Users/Shared/ccc/project/ai4
python world/tests/test_world.py

# world examples
python world/examples/frozen_lake_example.py
python world/examples/cartpole_example.py

# nn demo (uses uv)
cd nn && uv run chargpt_demo.py
```

## Installation

```bash
# world as a package
pip install -e world

# nn dependencies
pip install numpy
```

## Project Structure

```
ai4/
├── __init__.py          # from ai4 import world, nn
├── world/               # RL environment framework
│   ├── __init__.py
│   ├── core.py          # Env, StepResult
│   ├── envs/            # FrozenLakeEnv, CartPoleEnv
│   ├── spaces/           # Discrete, Box
│   ├── utils/            # registry, make, register
│   ├── wrappers/         # TimeLimitWrapper, RecordEpisodeWrapper
│   ├── examples/
│   └── tests/
│
└── nn/                  # Neural network framework
    ├── __init__.py
    ├── tensor.py         # Tensor with autodiff
    ├── nn.py             # Module, Linear, Embedding, RMSNorm, Adam
    ├── gpt.py            # GPT language model
    ├── chargpt.py        # training & inference
    └── chargpt_demo.py   # CharGPT demo script
```