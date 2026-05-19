# AGENTS.md

## Project Overview

Two independent Python subpackages under one repo:
- **world/** — Lightweight RL environment framework (pure Python + NumPy)
- **nn/** — DIY neural network framework (NumPy-based)

No monorepo tool. No `setup.py` / `pyproject.toml`. Each subpackage is standalone.

---

## world — Reinforcement Learning Environments

```bash
# Run all tests
python world/tests/test_world.py

# Install (if needed)
pip install numpy
pip install -e world

# Run examples
python world/examples/frozen_lake_example.py
python world/examples/cartpole_example.py
```

Built-in envs: `FrozenLake-v0`, `FrozenLake-v1`, `FrozenLake8x8-v1`, `CartPole-v1`
API: `world.make(id)`, `env.reset(seed=...)`, `env.step(action)` returns `StepResult` (tuple-unpackable).

---

## nn — Neural Network Framework

```bash
# Run CharGPT demo (downloads input.txt automatically on first run)
cd nn && uv run chargpt_demo.py
```

Uses `uv` (not plain `python`) to run scripts. Requires `numpy`.

---

## ml — Machine Learning Toolkit

```bash
# Run tests (requires pytest)
python -m pytest ml/tests/test_ml.py

# Run examples
PYTHONPATH=. python ml/examples/example.py
```

Modules: `linear_models`, `tree`, `ensemble`, `clustering`, `decomposition`, `metrics`, `preprocessing`
API: `from ml import LinearRegression`, `from ml.metrics import accuracy_score`, etc.

---

## Quirks / Gotchas

- **No package config** — No `setup.py`, `pyproject.toml`, or `requirements.txt` in repo root. Dependencies are installed ad-hoc.
- **input.txt auto-download** — `chargpt_demo.py` auto-downloads `names.txt` from GitHub if `input.txt` is missing.
- **world package structure** — The `world/` directory contains `world/` (the actual package). Use `pip install -e world` from the `world/` directory, or set `PYTHONPATH=/path/to/world` before importing.
- **Alternative nn demo run** — `PYTHONPATH=. uv run -m nn.chargpt_demo` from repo root also works.