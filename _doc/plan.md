# 重構計畫：整合 nn2kv 與 world 至 ai4py

## 目標

達成以下使用方式：
```python
from ai4py import nn, world    # 主要用法
import world; world.make(...)  # 向後相容
```

## 最終目錄結構

```
ai4py/                          # 頂層 Python 套件（需 __init__.py）
├── __init__.py                 # from ai4py import nn, world
│
├── world/                      # 子包 world/
│   ├── __init__.py
│   ├── core.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── frozen_lake.py
│   │   └── cartpole.py
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── discrete.py
│   │   └── box.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   └── random_agent.py
│   ├── wrappers/
│   │   ├── __init__.py
│   │   ├── time_limit.py
│   │   └── record_episode.py
│   ├── examples/
│   └── tests/
│
├── nn/                         # 子包 nn/
│   ├── __init__.py
│   ├── tensor.py
│   ├── nn.py
│   ├── gpt.py
│   ├── chargpt.py
│   ├── chargpt_demo.py
│   └── examples/
│
├── _doc/
│   └── plan.md
└── README.md
```

---

## 執行步驟

### 步驟 1：重構 world/ 為雙層子包

**現況**：`world/` 是專案根目錄，內含 `world/` 是 Python 套件
```
world/                 # 外層只是資料夾
├── world/             # 這才是 import world 的套件
├── examples/
├── tests/
└── README.md
```

**目標**：`world/` 成為 ai4py 下的子包
```
ai4py/world/           # 直接是 Python 套件
├── __init__.py
├── core.py            # 從 world/world/core.py 移上來
├── envs/              # 從 world/world/envs/ 移上來
├── spaces/            # 從 world/world/spaces/ 移上來
├── utils/            # 從 world/world/utils/ 移上來
├── wrappers/         # 從 world/world/wrappers/ 移上來
└── examples/
```

**執行**：
```bash
cd /Users/Shared/ccc/project/ai4py

# 將 world/world/ 的內容移到 world/（提升一層）
mv world/world/* world/
rmdir world/world

# 在 world/ 下加入 __init__.py（內容從原 world/world/__init__.py 讀取）
```

**world/__init__.py**（新建，內容與原本相同）：
```python
"""world - Lightweight RL environment framework"""
from world.core import Env, StepResult
from world.spaces import Discrete, Box
from world.envs import FrozenLakeEnv, CartPoleEnv
from world.wrappers import TimeLimitWrapper, RecordEpisodeWrapper
from world.utils import register, make, registry, run_random_agent

register("FrozenLake-v0",     FrozenLakeEnv, map_name="4x4", is_slippery=False)
register("FrozenLake-v1",     FrozenLakeEnv, map_name="4x4", is_slippery=True)
register("FrozenLake8x8-v1",  FrozenLakeEnv, map_name="8x8", is_slippery=True)
register("CartPole-v1",       CartPoleEnv,   max_steps=500)

__all__ = [
    "Env", "StepResult",
    "Discrete", "Box",
    "FrozenLakeEnv", "CartPoleEnv",
    "TimeLimitWrapper", "RecordEpisodeWrapper",
    "register", "make", "registry", "run_random_agent",
]
```

---

### 步驟 2：建立 nn/ 子包（從 nn2kv 而來）

```bash
mkdir -p nn/nn nn/examples
```

**複製並修改 nn2kv 檔案**：

| nn2kv 檔案 | 目的地 | 修改 |
|-----------|--------|------|
| `tensor.py` | `nn/nn/tensor.py` | 無需修改 |
| `nn.py` | `nn/nn/nn.py` | `from tensor import` → `from .tensor import` |
| `gpt.py` | `nn/nn/gpt.py` | 相對匯入 |
| `chargpt.py` | `nn/nn/chargpt.py` | 相對匯入 |
| `main.py` | `nn/nn/chargpt_demo.py` | 相對匯入，重新命名 |

**nn/nn/__init__.py**：
```python
"""nn - DIY Neural Network Framework (NumPy-based)"""
from .tensor import Tensor, cat
from .nn import Module, Linear, Embedding, RMSNorm, Adam
from .gpt import GPT
from .chargpt import train_model, generate_samples

__all__ = [
    "Tensor", "cat",
    "Module", "Linear", "Embedding", "RMSNorm", "Adam",
    "GPT",
    "train_model", "generate_samples",
]
```

---

### 步驟 3：建立 ai4py/__init__.py（頂層入口）

```python
"""ai4py - DIY AI Framework for Python

Subpackages
-----------
world : Reinforcement learning environments
nn    : Neural network framework (NumPy-based)

Examples
--------
>>> from ai4py import world, nn
>>> env = world.make("FrozenLake-v1")
"""

from . import world
from . import nn

__all__ = ["world", "nn"]
```

---

### 步驟 4：建立 ai4py/README.md

統一說明 ai4py 的兩個子包。

---

### 步驟 5：處理 world/ 內的 relative import

世界/中的模組（如 `envs/`、`utils/`、`wrappers/`）使用 `from world.xxx import` 語句。

將這些改為相對匯入：
```python
# 原本
from world.core import Env, StepResult
from world.spaces import Discrete, Box

# 改為
from .core import Env, StepResult
from .spaces import Discrete, Box
```

或使用 `world.` prefix（推薦，保持明確）：
```python
from world.core import ...
```

---

### 步驟 6：驗證

```bash
# 測試主要用法
cd /Users/Shared/ccc/project/ai4py
python -c "from ai4py import nn, world; print(world.__name__, nn.__name__)"

# 測量子包獨立使用（向後相容）
python -c "import world; env = world.make('FrozenLake-v1')"

# 測試 nn
cd ai4py/nn/nn && uv run chargpt_demo.py

# 測試 world
cd ai4py/world && python tests/test_world.py
```

---

## nn2kv import 修改對照

| 檔案 | 原本 | 修改後 |
|------|------|--------|
| `nn.py` | `from tensor import Tensor` | `from .tensor import Tensor` |
| `gpt.py` | `from tensor import Tensor, cat`<br>`from nn import Module, ...` | `from .tensor import Tensor, cat`<br>`from .nn import Module, ...` |
| `chargpt.py` | `from tensor import Tensor`<br>`from nn import Adam`<br>`from gpt import GPT` | `from .tensor import Tensor`<br>`from .nn import Adam`<br>`from .gpt import GPT` |
| `chargpt_demo.py` | `from nn import Adam`<br>`from chargpt import ...` | `from .nn import Adam`<br>`from .chargpt import ...` |

---

## 注意事項

1. **雙層 vs 三層** — 此結構是 `ai4py/world/` 和 `ai4py/nn/` 雙層，無需 `ai4py/world/world/` 的第三層。

2. **world/ 內部引用** — `world/core.py`、`world/envs/__init__.py` 等之間的 import 需要改為相對匯入（如 `from .spaces import Discrete`）或保持 `world.` prefix。

3. **安裝方式** — 使用者安裝 ai4py 後，無需知道內部結構，直接 `from ai4py import world, nn`。

4. **input.txt** — `chargpt_demo.py` 會在缺少時自動下載，不隨套件分發。

## 尚未完成項目

- [ ] 重構 world/ 目錄（提升一層，移除多餘的 world/world/）
- [ ] 建立 world/__init__.py
- [ ] 建立 nn/nn/結構成 nn2kv 檔案
- [ ] 修改 nn2kv 的 import 為相對匯入
- [ ] 建立 ai4py/__init__.py
- [ ] 處理 world/ 內部 import
- [ ] 建立統一的 ai4py/README.md
- [ ] 驗證所有用法