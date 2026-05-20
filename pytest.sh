#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"

pytest world/tests
pytest nn/tests
pytest ml/tests
