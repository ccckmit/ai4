#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"

echo "=== world/frozen_lake_example ==="
uv run python world/examples/frozen_lake_example.py

echo ""
echo "=== world/cartpole_example ==="
uv run python world/examples/cartpole_example.py

echo ""
echo "=== world/frozenlake_qtable ==="
uv run python world/examples/frozenlake_qtable.py

echo ""
echo "=== ml/example ==="
uv run python ml/examples/example.py

echo ""
echo "=== nn/chargpt_demo ==="
uv run python -m nn.chargpt_demo