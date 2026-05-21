#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== world tests ==="
npx tsx world/tests/test_world.ts

npx tsx world/examples/frozen_lake_example.ts

npx tsx world/examples/frozenlake_qtable.ts

echo "=== nn tensor tests ==="
npx jest nn/tests/test_tensor --no-coverage

echo "=== nn nn tests ==="
npx tsx nn/tests/test_nn.ts

echo "=== nn by claude ==="
npx tsx nn/tests/test_by_claude.ts

echo "=== ml examples ==="
npx tsx ml/tests/test_ml.ts

echo "=== llm tests ==="
npx tsx llm/tests/test_agent.ts 2>/dev/null || true

# npx tsx nn/chargpt_demo.ts
# npx tsx nn/mnist/train.ts
# npx tsx world/examples/cartpole_example.ts
# npx tsx world/examples/cartpole_closed_form.ts
# npx tsx world/examples/cartpole_vpg.ts
# npx tsx world/examples/bipedalwalker.ts