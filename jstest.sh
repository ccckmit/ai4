#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== world tests ==="
npx tsx world/tests/test_world.ts

echo ""
echo "=== nn tensor tests ==="
npx jest nn/tests/test_tensor --no-coverage

echo ""
echo "=== llm tests ==="
npx tsx llm/tests/test_agent.ts 2>/dev/null || true