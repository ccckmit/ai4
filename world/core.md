# world/core.md - 環境框架核心理論

本模組定義了 RL 環境的基礎抽象介面，包括 `Env` 類別（所有環境的基底類）和 `StepResult` 資料類別（封裝 `step()` 的返回值）。

## OpenAI Gym 風格介面

本框架的環境介面參考了 OpenAI Gym 的設計，是 RL 研究的事實標準：

```python
env.reset(seed=42)    # 初始化環境，返回初始觀測
env.step(action)      # 執行動作，返回 StepResult
env.render()          # 視覺化環境可選
env.close()           # 清理資源
```

統一的介面讓同一套 RL 演算法可以無縫切換不同環境，促進了 RL 演算法的快速發展。

## StepResult 資料類別

`step()` 方法返回一個 `StepResult` 物件，封裝了五個關鍵資訊：

| 欄位 | 類型 | 說明 |
|------|------|------|
| `observation` | ObsType | 當前狀態 |
| `reward` | float | 立即獎勵 |
| `terminated` | bool | 是否達到終止狀態（成功或失敗） |
| `truncated` | bool | 是否被人為截斷（如時間限制） |
| `info` | dict | 附加診斷資訊 |

`done` 屬性是 `terminated or truncated`，作為回合結束的通用判斷。

為何要區分 `terminated` 和 `truncated`？
- `terminated`：真正的 MDP 終止訊號（如智慧體掉入洞穴）
- `truncated`：環境強制的截斷（如最大步數限制），不代表 MDP 的終止

這種區分在稀疏獎勵場景中很重要——TRUNCATED 的回合不代表任務失敗，agent 可能只是沒能在限制步數內完成。

## Env 抽象類別

`Env` 是一個 abstract base class，強制子類實現以下介面：

- `observation_space`：描述有效觀測的 Space 物件
- `action_space`：描述有效動作的 Space 物件  
- `reset(seed=...)`：初始化，返回 `(obs, info)`
- `step(action)`：執行動作，返回 StepResult

可選覆蓋：
- `render()`：視覺化
- `close()`：資源清理
- `seed()`：設定隨機種子

## 裝飾器模式：Wrapper

Wrapper 是一種將環境功能進行增強的設計模式：

```python
wrapped_env = TimeLimitWrapper(env, max_steps=200)
wrapped_env = RecordEpisodeWrapper(wrapped_env)
```

每個 wrapper 包裝一個現有環境，在不改變底層環境的情況下新增功能：
- `TimeLimitWrapper`：限制每回合最大步數
- `RecordEpisodeWrapper`：記錄並統計每回合的獎勵

這種組合式設計比繼承更靈活——可以任意組合多個 wrapper。

## 馬可夫性質

本框架假設環境滿足**馬可夫性質**：轉移機制 $P(s'|s,a)$ 只依賴當前狀態和動作，不依賴歷史。也就是說，「現在，未來與過去無關，只與現在有關」。

這是強化學習理論的基礎。實務上，大多數 RL 環境（如 Atari 遊戲、CartPole）都近似滿足這個性質。

## 隨機種子設定

確定的環境行為對研究至關重要：

```python
env = world.make("FrozenLake-v1")
obs, info = env.reset(seed=42)  # 可重現的初始化
```

內部使用 `np.random.default_rng()` 產生確定性的隨機數流。

## StepResult 的迭代協定

`StepResult` 實現了 `__iter__`，支援元組解包：

```python
obs, reward, terminated, truncated, info = env.step(action)
```

這與 Gym 的設計一致，允許快速使用元組 unpacking 取得各欄位。

---

**相關連結**：[FrozenLake.md](envs/frozen_lake.md) | [CartPole.md](envs/cartpole.md)