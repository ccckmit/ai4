# AGENTS.md

## 套件結構

4 個獨立子套件（純 Python + NumPy，無外部 ML 框架依賴）：

| 目錄 | 說明 | 主要模組 |
|------|------|----------|
| `world/` | RL 環境框架（Gym 風格） | `core.py`, `envs/`, `spaces/`, `wrappers/`, `utils/` |
| `nn/` | DIY 深度學習框架（自動微分） | `tensor.py`, `nn.py`, `gpt.py`, `cnn.py`, `datasets.py` |
| `ml/` | 機器學習工具箱 | `linear_models`, `tree`, `ensemble`, `clustering`, `decomposition`, `metrics`, `preprocessing` |
| `llm/` | LLM 代理框架（Python + TypeScript + Rust） | `agent.py`, `agent.ts`, `agent.rs` |

**重要：`ml` 和 `llm` 不是 `ai4` 的子模組。** 只能透過 `PYTHONPATH=. import ml` 或单独安装使用。

## 匯入方式

### world / nn（可安裝為 pip 套件）
```python
# 方式 1：PYTHONPATH（開發用）
import world
env = world.make("FrozenLake-v1")

# 方式 2：pip 安裝（需先 `pip install -e .`）
from ai4 import world
env = world.make("FrozenLake-v1")
```

### ml / llm（僅 PYTHONPATH）
```bash
export PYTHONPATH=/path/to/ai4
```
```python
import ml
from ml import LinearRegression
import llm
from llm import call_ollama
```

## 測試

```bash
# Python（uv + pytest）
./pytest.sh                    # 全部測試
uv run pytest world/tests       # 單一模組
uv run pytest nn/tests
uv run pytest ml/tests

# TypeScript（npx tsx）
./jstest.sh                    # 全部測試（需在 repo 根目錄）
npx tsx world/tests/test_world.ts
npx tsx nn/tests/test_nn.ts
npx tsx nn/tests/test_tensor.ts
npx tsx nn/tests/test_by_claude.ts
npx tsx nn/tests/test_gpt.ts    # 可選（有 GPU 需求）
npx tsx ml/tests/test_ml.ts
npx tsx llm/tests/test_agent.ts # 可選（需要 ollama）
```

## 重要約定

- **uv** — 所有 Python 腳本使用 `uv run python` 或 `uv run pytest`
- **PYTHONPATH** — `pytest.sh` 自動設定；獨立執行時需手動設為 repo 根目錄
- **TypeScript** — 使用 `npx tsx` 執行 `.ts` 檔案（非 node/jest）
- **world.step() 回傳** — `StepResult` 可解包為 5 元組：`obs, reward, terminated, truncated, info`
- **Python 程式碼註解** — 使用英文；理論文件使用繁體中文

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

- `README.md` / `world/README.md` / `nn/README.md` / `ml/README.md`
- `\_wiki/` — 知識庫理論文章（繁體中文）
- `nn/tensor.md`, `nn/gpt.md`, `world/core.md`, `ml/linear_models.md`