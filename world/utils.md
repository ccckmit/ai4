# world/utils.md - 註冊表與工廠模式

本模組提供了環境的**註冊機制**和**工廠函式**，讓使用者可以透過字串名稱來建立環境，而不用直接 import 具體的環境類別。

## 註冊表（Registry）模式

註冊表是一個全域字典，將環境名稱映射到環境類別與其初始化參數：

```python
registry = {
    "FrozenLake-v0": (FrozenLakeEnv, {"map_name": "4x4", "is_slippery": False}),
    "FrozenLake-v1": (FrozenLakeEnv, {"map_name": "4x4", "is_slippery": True}),
    "CartPole-v1":   (CartPoleEnv,   {"max_steps": 500}),
    ...
}
```

### 為什麼需要註冊表

1. **鬆散耦合**：使用者不需要知道具體類別名稱和 import 路徑
2. **統一的建立介面**：所有環境用 `world.make("name")` 建立
3. **版本管理**：同名環境可以有多個版本（如 FrozenLake-v0、v1）
4. **可擴展**：第三方可以註冊自訂環境

## register() 與 make()

### register（註冊）

```python
register("FrozenLake-v0", FrozenLakeEnv, map_name="4x4", is_slippery=False)
```

- 第一個參數是環境名稱（字串 ID）
- 第二個參數是環境類別
- 其餘參數是傳遞給類別建構式的關鍵字參數

### make（工廠）

```python
env = make("FrozenLake-v1")
```

`make()` 從註冊表查找名稱，用儲存的參數建立環境實例：

```python
def make(id):
    cls, kwargs = registry[id]
    return TimeLimitWrapper(cls(**kwargs))
```

Factory Method 模式讓建立過程統一，且可以自動套用預設的 Wrapper。

## 內建環境註冊

每次 import `world` 時，自動註冊以下環境：

| 名稱 | 類別 | 說明 |
|------|------|------|
| `FrozenLake-v0` | FrozenLakeEnv | 4×4，不滑動 |
| `FrozenLake-v1` | FrozenLakeEnv | 4×4，滑動（標準） |
| `FrozenLake8x8-v1` | FrozenLakeEnv | 8×8，滑動 |
| `CartPole-v1` | CartPoleEnv | 標準平衡問題 |
| `BipedalWalker-v3` | BipedalWalkerEnv | 雙足行走（需 pygame） |

## 隨機 Agent（random_agent）

`run_random_agent()` 是一個快速的環境測試工具：

```python
run_random_agent(env, episodes=3, render=True)
```

每次動作從 `env.action_space.sample()` 隨機取樣。用途：
- 驗證環境是否正常運作
- 了解動作空間的形狀
- 作為基線（baseline）效能參考

---

**相關連結**：[core.md](core.md) | [envs.md](envs.md)
