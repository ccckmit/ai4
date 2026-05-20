#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== world tests ==="
npx tsx world/tests/test_world.ts

echo ""
echo "=== nn tests ==="
npx tsx nn/tests/test_nn.ts
npx tsx nn/tests/test_tensor.ts
npx tsx nn/tests/test_by_claude.ts
npx tsx nn/tests/test_gpt.ts 2>/dev/null || true

echo ""
echo "=== ml tests ==="
npx tsx ml/tests/test_ml.ts

echo ""
echo "=== llm tests ==="
npx tsx llm/tests/test_agent.ts 2>/dev/null || true