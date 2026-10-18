#!/usr/bin/env bash
# 依序執行所有範例；任何一個失敗就停止。
# 用法：./run.sh            執行全部
#       ./run.sh gpt gru    只執行指定範例（gpt gru mlp-mnist cnn-mnist）
set -x
set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-python3}"
examples=("$@")
[ ${#examples[@]} -eq 0 ] && examples=(gpt gru mlp-mnist cnn-mnist)

for name in "${examples[@]}"; do
    if [ ! -f "$name.py" ]; then
        echo "找不到範例：$name.py" >&2
        exit 1
    fi
    echo "=================== $name ==================="
    "$PYTHON" "$name.py"
    echo
done
echo "全部完成。"