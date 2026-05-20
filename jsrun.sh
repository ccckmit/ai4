#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== ml/example ==="
npx tsx ml/examples/example.ts

echo ""
echo "=== world/frozen_lake_example ==="
npx tsx world/examples/frozen_lake_example.ts

echo ""
echo "=== world/cartpole_example ==="
npx tsx world/examples/cartpole_example.ts

echo ""
echo "=== world/frozenlake_qtable ==="
npx tsx world/examples/frozenlake_qtable.ts

echo ""
echo "=== nn/chargpt_demo ==="
npx tsx nn/chargpt_demo.ts