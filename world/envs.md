# world/envs.md - 環境實作理論

本模組實現了三種強化學習環境：FrozenLake、CartPole 和 BipedalWalker。它們遵循 `world/core.md` 定義的統一 `Env` 介面，但各有不同的狀態空間、動作空間與轉移動態。

## FrozenLake（柵格世界）

FrozenLake 是一個經典的**柵格世界（grid world）** 問題，靈感來自強化學習經典《Sutton & Barto》。

### 環境設定

4×4 或 8×8 的柵格，agent 需要從起點 S 走到終點 G，避開洞穴 H：

```
S  F  F  F
F  H  F  H
F  F  F  H
H  F  F  G
```

- **觀測（observation）**：離散值 $[0, 15]$（4×4 的展平索引），表示 agent 當前所在格子
- **動作（action）**：0=左、1=下、2=右、3=上
- **獎勵（reward）**：到達終點 G 得 +1，其他地方 0
- **終止條件**：掉入洞穴 H 或到達終點 G

### 滑動機制（Slippery）

FrozenLake 的核心難點是**非確定性的轉移**：
- `is_slippery=True`（預設）：agent 有 1/3 機率往目標方向走，1/3 機率偏移到垂直方向
- `is_slippery=False`：確定性環境，適合測試基礎規劃

這讓 FrozenLake 成為測試**無模型（model-free）RL 演算法**（如 Q-Learning）的標準環境。

詳細理論請見 [_wiki/FrozenLake-Environment.md](../_wiki/FrozenLake-Environment.md)。

## CartPole（倒單擺）

CartPole 是經典的**連續控制**問題，也是 OpenAI Gym 中最廣泛使用的入門環境。

### 問題描述

一個小車在軌道上滑動，頂端連接一根垂直桿子。目標是透過左右移動小車，保持桿子直立。

- **觀測**：4 維連續向量 $(\text{pos}, \dot{\text{pos}}, \text{angle}, \dot{\text{angle}})$
- **動作**：離散值 0=向左推、1=向右推
- **獎勵**：每步 +1（只要桿子還直立）
- **終止條件**：桿子傾斜超過 ±12°、小車超出邊界、或達最大步數

### 物理模型

CartPole 的轉移動態由 Lagrangian 力學推導：
$$
\ddot{\theta} = \frac{g \sin\theta - \alpha m l \dot{\theta}^2 \sin(2\theta)/2 - \alpha \cos\theta \cdot F}{4l/3 - \alpha m l \cos^2\theta}
$$
其中 $\alpha = 1/(m + M)$，$m$ 是桿子質量，$M$ 是小車質量，$F$ 是施加的力。

## BipedalWalker（雙足行走）

BipedalWalker 是較複雜的**連續控制環境**，agent 需要控制一個雙足機器人行走。

- **觀測**：24 維連續向量（關節角度、速度、接觸感測器等）
- **動作**：4 維連續向量 $[-1, 1]$（髖關節與膝關節力矩）
- **獎勵**：向前移動獎勵 - 能量消耗 - 摔落懲罰
- **終止條件**：機身傾倒或達到最大步數

BipedalWalker 是 Box2D 模擬環境，需要的策略比 FrozenLake 和 CartPole 複雜得多。詳細理論請見 [_wiki/BipedalWalker.md](../_wiki/BipedalWalker.md)。

## 三種環境的比較

| 特性 | FrozenLake | CartPole | BipedalWalker |
|------|-----------|----------|---------------|
| 觀測類型 | 離散（單整數） | 連續（4 維） | 連續（24 維） |
| 動作類型 | 離散（4 方向） | 離散（2 方向） | 連續（4 維） |
| 轉移確定性 | 隨機 | 確定性 | 確定性 |
| Problem 類型 | 導航/規劃 | 平衡控制 | 行走控制 |
| 難度 | 簡單 | 中等 | 困難 |

---

**相關連結**：[core.md](core.md) | [Reinforcement-Learning.md](../_wiki/Reinforcement-Learning.md)
