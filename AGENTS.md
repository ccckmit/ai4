# AGENTS.md

## 專案概覽

四個獨立子套件（純 Python + NumPy，無外部 ML 框架依賴）：

| 目錄 | 說明 | 主要模組 |
|------|------|----------|
| `world/` | RL 環境框架（Gym 風格） | `core.py`, `envs/`, `spaces/`, `wrappers/`, `utils/` |
| `nn/` | DIY 深度學習框架（自動微分） | `tensor.py`, `nn.py`, `gpt.py`, `cnn.py`, `datasets.py` |
| `ml/` | 機器學習工具箱 | `linear_models`, `tree`, `ensemble`, `clustering`, `decomposition`, `metrics`, `preprocessing` |
| `llm/` | LLM 代理框架（Python + TypeScript + Rust） | `agent.py`, `agent.ts`, `agent.rs` |

另有對應的 TypeScript 實作（`*.ts`），使用 `npx tsx` 執行。

## 測試

```bash
# Python（uv + pytest）
./pytest.sh                    # 全部測試
uv run pytest world/tests       # 單一模組
uv run pytest nn/tests
uv run pytest ml/tests

# TypeScript（npx tsx）
./jstest.sh                    # 全部測試
npx tsx world/tests/test_world.ts
npx tsx nn/tests/test_nn.ts
npx tsx nn/tests/test_tensor.ts
npx tsx nn/tests/test_by_claude.ts
npx tsx nn/tests/test_gpt.ts    # 可選（有 GPU 需求）
npx tsx ml/tests/test_ml.ts
npx tsx llm/tests/test_agent.ts # 可選（需要 ollama）
```

## 世界（world）— RL 環境

```python
# 方式 1：從根目錄執行（需設定 PYTHONPATH）
import world
env = world.make("FrozenLake-v1")

# 方式 2：當作 pip 安裝的套件
from ai4 import world
env = world.make("FrozenLake-v1")

obs, info = env.reset(seed=42)
result = env.step(action)
obs, reward, terminated, truncated, info = result  # StepResult 可解包
```

內建環境：`FrozenLake-v0`（確定性）、`FrozenLake-v1`（隨機滑動）、`FrozenLake8x8-v1`、`CartPole-v1`、`BipedalWalker-v3`

## 類神經網路（nn）— DIY 深度學習框架

```python
from nn import Tensor, cat, Module, Linear, Embedding, RMSNorm, Adam, GPT

a = Tensor([[1, 2]], requires_grad=True)
b = Tensor([[3, 4]], requires_grad=True)
c = a @ b
loss = c.sum()
loss.backward()
print(a.grad)
```

訓練範例：
```bash
uv run python -m nn.chargpt_demo   # 訓練 CharGPT 生成中文名字（自動下載 names.txt）
```

## ML 工具箱

```python
from ml import LinearRegression, LogisticRegression, DecisionTree
from ml.metrics import accuracy_score

model = LinearRegression(lr=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

範例：`PYTHONPATH=. uv run python ml/examples/example.py`

## LLM 代理框架

```python
from llm import call_ollama, build_context, conversation_history
response = call_ollama("What is 2+2?")
```

## 重要約定

- **uv** — 所有 Python 腳本使用 `uv run python` 或 `uv run pytest`
- **PYTHONPATH** — `pytest.sh` 自動設定 `PYTHONPATH=.`；獨立執行時需手動設定
- **TypeScript** — 使用 `npx tsx` 執行 `.ts` 檔案（非 node）
- **Python 程式碼註解** — 使用**英文**（方便自動生成英文文件）；理論文件使用**繁體中文**

## Python ↔ TypeScript 測試對照

| Python | TypeScript |
|---------|------------|
| `world/tests/test_world.py` | `world/tests/test_world.ts` |
| `nn/tests/test_nn.py` | `nn/tests/test_nn.ts` |
| `nn/tests/test_tensor.py` | `nn/tests/test_tensor.ts` |
| `nn/tests/test_by_claude.py` | `nn/tests/test_by_claude.ts` |
| `nn/tests/test_gpt.py` | `nn/tests/test_gpt.ts` |
| `ml/tests/test_ml.py` | `ml/tests/test_ml.ts` |
| `llm/tests/test_agent.py` | `llm/tests/test_agent.ts` |

## 文檔位置

- `README.md` / `world/README.md` / `nn/README.md` / `ml/README.md` — 各子套件使用說明
- `\_wiki/` — 知識庫理論文章（繁體中文）：Q-Learning, Reinforcement-Learning, Backpropagation, Transformer, Attention, RMSNorm, GPT, Gradient-Descent
- `nn/tensor.md` — 自動微分理論
- `nn/gpt.md` — GPT 模型理論
- `world/core.md` — 環境框架理論
- `ml/linear_models.md` — 線性模型理論