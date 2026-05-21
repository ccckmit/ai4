# world/spaces.md - 空間理論

在強化學習中，**空間（Space）** 定義了環境中觀測和動作的合法取值範圍。本模組實現了兩種最常見的空間類型：`Discrete`（離散空間）和 `Box`（連續空間）。

## 空間的角色

每個 `Env` 都有 `observation_space` 和 `action_space` 屬性：

```python
env = world.make("CartPole-v1")
print(env.observation_space)  # Box(4,) — 4 維連續
print(env.action_space)       # Discrete(2) — 2 個離散動作
```

空間的作用：
1. **定義合法範圍**：agent 可以據此選擇動作
2. **提供取樣**：`space.sample()` 隨機取樣，用於隨機 agent 測試
3. **包含檢查**：`space.contains(x)` 驗證值是否合法

## Discrete（離散空間）

離散空間表示一個有限集合 $\\{0, 1, ..., n-1\\}$：

```python
space = Discrete(4)  # 四個動作：左、下、右、上
```

- `sample()`：均勻隨機返回 $0, ..., n-1$ 中的一個
- `contains(x)`：檢查 $0 \le x < n$ 且 x 為整數

FrozenLake 的動作空間就是 `Discrete(4)`（四個方向），觀測空間也是 `Discrete(16)`（16 個格子）。

## Box（連續空間）

Box 空間表示 n 維連續空間中的矩形區域：

```python
space = Box(low=-1, high=1, shape=(4,))
```

- `low` / `high`：可以是純量（廣播到各維度）或陣列（逐維邊界）
- `sample()`：在範圍內均勻隨機取樣
- `contains(x)`：檢查所有維度都在 $[low_i, high_i]$ 範圍內

### CartPole 的 Box 空間

```python
Box(low=np.array([-4.8, -inf, -0.418, -inf]),
    high=np.array([4.8, inf, 0.418, inf]))
```

四個維度分別是：小車位置、小車速度、桿子角度、桿子角速度。位置和角度有明確邊界，速度和角速度理論上無界（實務上截斷）。

### BipedalWalker 的 Box 空間

觀測空間 `Box(24,)` 和動作空間 `Box(4, low=-1, high=1)` 都是連續空間。24 維觀測包含關節角度、速度、接觸感測器以及雷射測距儀資料。

## 空間作為 API 合約

空間不僅是描述性的，還是可程式化的合約：

1. **策略約束**：策略網路的輸出層必須匹配 `action_space` 的維度
2. **規格化輸入**：`observation_space` 告訴我們是否需要對輸入做歸一化
3. **測試輔助**：`space.sample()` 讓隨機策略測試可以自動化

隨機 agent 的實作就利用了這個特性：

```python
action = env.action_space.sample()  # 無需知道具體空間類型
```

---

**相關連結**：[core.md](core.md) | [envs.md](envs.md) | [Reinforcement-Learning.md](../_wiki/Reinforcement-Learning.md)
