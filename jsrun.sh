#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== world examples ==="
npx tsx world/examples/frozen_lake_example.ts
npx tsx world/examples/frozenlake_qtable.ts
npx tsx world/examples/cartpole_example.ts
npx tsx world/examples/cartpole_closed_form.ts
#npx tsx world/examples/cartpole_closed_form.ts --render
#npx tsx world/examples/cartpole_vpg.ts

echo ""
echo "=== ml examples ==="
npx tsx ml/examples/example.ts

echo ""
echo "=== nn examples ==="
npx tsx nn/chargpt_demo.ts

echo ""
echo "=== llm agent test ==="
npx tsx llm/tests/test_agent.ts
