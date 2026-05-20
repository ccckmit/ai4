#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"

uv run python world/examples/frozen_lake_example.py

uv run python world/examples/cartpole_example.py

uv run python world/examples/frozenlake_qtable.py

uv run python ml/examples/example.py

uv run python -m nn.chargpt_demo