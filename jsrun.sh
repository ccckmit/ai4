#!/bin/bash
set -x

cd "$(dirname "$0")"

echo "=== world examples ==="
npx tsx world/examples/frozen_lake_example.ts
npx tsx world/examples/frozenlake_qtable.ts

echo ""
echo "=== llm agent test ==="
npx tsx llm/tests/test_agent.ts