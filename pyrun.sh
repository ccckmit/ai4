#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"

python world/examples/frozen_lake_example.py

python world/examples/cartpole_example.py

python world/examples/frozenlake_qtable.py

python ml/examples/example.py

python -m nn.chargpt_demo

python nn/mnist/train.py

# python world/examples/cartpole_closed_form.py
python world/examples/cartpole_closed_form.py --render

# python world/examples/cartpole_vpg.py

# python world/examples/bipedalwalker.py