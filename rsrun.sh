#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== frozen_lake_example ==="
cargo run --bin world_frozen_lake_example

echo ""
echo "=== frozenlake_qtable ==="
cargo run --bin world_frozenlake_qtable

echo ""
echo "=== cartpole_closed_form ==="
cargo run --bin world_cartpole_closed_form
# cargo run --bin world_cartpole_render_test

echo ""
echo "=== ml example ==="
cargo run --bin ml_example

echo ""
echo "=== nn example ==="
cargo run --bin nn_example

echo ""
echo "=== chargpt_demo ==="
cargo run --bin chargpt_demo

echo ""
echo "=== mnist_train ==="
cargo run --bin mnist_train