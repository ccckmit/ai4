#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"
uv run pytest world/tests
uv run pytest nn/tests
uv run pytest ml/tests
