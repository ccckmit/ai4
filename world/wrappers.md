# world/wrappers.md - 包裝器理論

本模組實現了環境包裝器（Wrapper），這是強化學習框架中的標準設計模式，用於在不修改環境原始程式碼的前提下擴充功能。

## 裝飾器模式

Wrapper 採用**裝飾器模式（Decorator Pattern）**：包裝一個 `Env` 物件，攔截其方法呼叫，新增或修改行為後再轉發給底層環境。

```python
env = FrozenLakeEnv()                     # 原始環境
env = TimeLimitWrapper(env, max_steps=100)  # 加上時間限制
env = RecordEpisodeWrapper(env)            # 加上記錄功能
```

每個 Wrapper 都實作與 `Env` 相同的介面，所以可以任意疊加。

## TimeLimitWrapper（時間限制包裝器）

強化學習中，有些環境沒有自然終止條件（如 CartPole 理論上可以永遠平衡）。`TimeLimitWrapper` 為這些環境加上回合最大步數限制。

### 截斷的意義

當步數超過 `max_steps` 時，Wrapper 將 `truncated` 設為 True：

- **不是任務失敗**：agent 只是沒能在步數限制內達到目標
- **不同於 terminated**：真正的 MDP 終止（成功或死亡）設 `terminated=True`
- **Bellman 方程不變**：截斷不影響 MDP 的理論框架

### 實作方式

```python
class TimeLimitWrapper(Wrapper):
    def step(self, action):
        result = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_steps:
            result = result._replace(truncated=True)
        return result
```

原生的 `step()` 返回後，檢查是否超時，若超時則修改 `truncated` 標誌。

## RecordEpisodeWrapper（回合記錄包裝器）

用於監控和記錄整回合的統計資料，對訓練過程除錯和評估非常重要。

### 功能

1. **累積獎勵追蹤**：記錄每回合的總獎勵
2. **回合自動監控**：當 `terminated` 或 `truncated` 發生時，記錄完整回合
3. **回合統計輸出**：可輸出 `Episode 10: reward = 23.5, length = 200`

### 應用

- 訓練時監控收斂進度
- 比較不同超參數的效果
- 產生訓練曲線資料

## Wrapper 組合層疊

多個 Wrapper 的疊加順序很重要：

```python
env = FrozenLakeEnv()
env = RecordEpisodeWrapper(env)      # 層 2：記錄
env = TimeLimitWrapper(env, 200)     # 層 1：時間限制
```

從 agent 視角呼叫 `env.step()`：
1. 先經過 `TimeLimitWrapper.step()` 檢查時間
2. 再經過 `RecordEpisodeWrapper.step()` 記錄
3. 最後到達底層 `FrozenLakeEnv.step()`

這種分層設計讓每個 Wrapper 職責單一，可以自由組合。

---

**相關連結**：[core.md](core.md) | [envs.md](envs.md)
