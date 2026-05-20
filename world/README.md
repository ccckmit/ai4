# world/ — 強化學習環境框架

一個輕量級的 RL 環境框架，使用純 Python + NumPy 實現，接口設計參考 OpenAI Gym。

## 內建模境

| ID | 說明 |
|----|------|
| `FrozenLake-v0` | 4×4 冰湖（確定性，無滑動） |
| `FrozenLake-v1` | 4×4 冰湖（隨機滑動） |
| `FrozenLake8x8-v1` | 8×8 冰湖（更具挑戰） |
| `CartPole-v1` | 桿平衡控制 |

## 核心 API

```python
import world

env = world.make("FrozenLake-v1")       # 建立環境
obs, info = env.reset(seed=42)          # 初始化
result = env.step(action)               # 執行動作
# result = (observation, reward, terminated, truncated, info)

done = result.terminated or result.truncated  # 回合結束？
env.close()
```

## 包裝器（Wrapper）

- `TimeLimitWrapper(env, max_steps)`：限制每回合最大步數
- `RecordEpisodeWrapper(env)`：記錄並統計每回合獎勵

## 空間類型

- `Discrete(n)`：n 個離散動作/狀態
- `Box(shape)`：連續向量空間

## 範例

```bash
PYTHONPATH=. uv run python world/examples/frozen_lake_example.py
PYTHONPATH=. uv run python world/examples/cartpole_example.py
PYTHONPATH=. uv run python world/examples/frozenlake_qtable.py
```

## 模組結構

```
world/
  core.py         # Env, StepResult 基類
  spaces/         # Discrete, Box 空間定義
  envs/           # FrozenLake, CartPole 環境實現
  wrappers/       # TimeLimit, RecordEpisode 包裝器
  utils/          # 註冊表、隨機 agent
```

## 理論背景

詳見 [\_wiki/Reinforcement-Learning.md](../_wiki/Reinforcement-Learning.md) 和 [\_wiki/Q-Learning.md](../_wiki/Q-Learning.md)。