# AGENTS.md

## 專案概覽

三個獨立的 Python 子套件在同一個 repo 中：
- **world/** — 輕量級 RL 環境框架（純 Python + NumPy）
- **nn/** — DIY 深度學習框架（NumPy 自動微分 + Transformer）
- **ml/** — 機器學習工具箱（線性模型、決策樹、集成、聚類、PCA 等）

## 測試與執行

```bash
# 執行所有測試（使用 uv + pytest）
./test.sh

# 執行所有範例
./run.sh
```

## 世界（World）— 強化學習環境

### 執行測試
```bash
uv run pytest world/tests
```

### 執行範例
```bash
PYTHONPATH=. uv run python world/examples/frozen_lake_example.py   # Q-Learning
PYTHONPATH=. uv run python world/examples/cartpole_example.py       # PD 控制器
PYTHONPATH=. uv run python world/examples/frozenlake_qtable.py      # SARSA, TD(λ)
```

### API
```python
import world
env = world.make("FrozenLake-v1")
obs, info = env.reset(seed=42)
result = env.step(action)
obs, reward, terminated, truncated, info = result  # 可解包
```

內建環境：`FrozenLake-v0`, `FrozenLake-v1`, `FrozenLake8x8-v1`, `CartPole-v1`

## 類神經網路（nn）— DIY 深度學習框架

### 執行測試
```bash
uv run pytest nn/tests
```

### 執行範例
```bash
uv run python -m nn.chargpt_demo   # 訓練 CharGPT 生成中文名字
```

### API
```python
from nn import Tensor, cat, Module, Linear, Embedding, RMSNorm, Adam, GPT

# 手動建立神經網路
a = Tensor([[1, 2]], requires_grad=True)
b = Tensor([[3, 4]], requires_grad=True)
c = a @ b
loss = c.sum()
loss.backward()
print(a.grad)

# 建立 GPT 模型
model = GPT(vocab_size=100, block_size=32, n_layer=2, n_embd=32, n_head=4)
logits, caches = model(token_ids, kv_caches=None)
```

## ML 工具箱

### 執行測試
```bash
uv run pytest ml/tests
```

### 執行範例
```bash
PYTHONPATH=. uv run python ml/examples/example.py
```

### API
```python
from ml import LinearRegression, LogisticRegression, DecisionTree
from ml.metrics import accuracy_score

model = LinearRegression(lr=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

模組：`linear_models`, `tree`, `ensemble`, `clustering`, `decomposition`, `metrics`, `preprocessing`

## 特殊約定

- **使用 uv** — 所有腳本使用 `uv run python` 或 `uv run pytest`
- **PYTHONPATH** — `test.sh` 和 `run.sh` 自動設定 `PYTHONPATH=.`
- **input.txt 自動下載** — `chargpt_demo.py` 會自動下載 `names.txt`

## 測試腳本約定

```bash
#!/bin/bash
set -x

export PYTHONPATH="$(dirname "$0")"

uv run pytest world/tests
uv run pytest nn/tests
uv run pytest ml/tests
```

## 文檔位置

- **\_wiki/** — 知識庫概念文章（繁體中文，~300 行/篇）
  - Q-Learning.md, Reinforcement-Learning.md, Backpropagation.md
  - Transformer.md, Attention-Mechanism.md, RMSNorm.md, GPT.md, Gradient-Descent.md
- **nn/tensor.md** — 自動微分理論
- **nn/gpt.md** — GPT 模型理論
- **world/core.md** — 環境框架理論
- **ml/linear_models.md** — 線性模型理論
- **world/README.md, nn/README.md, ml/README.md** — 各子套件使用說明

## 程式碼註解

所有 Python 原始碼使用**英文註解**（方便自動生成英文文件）。理論背景文件則使用**繁體中文**。