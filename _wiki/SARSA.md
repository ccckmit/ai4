# SARSA（SARSA演算法）

SARSA 的全稱是 **State-Action-Reward-State-Action**（狀態-動作-獎勵-狀態-動作），是一種**在策略（on-policy）**時間差分（Temporal Difference, TD）控制演算法。與 Q-Learning 不同，SARSA 使用實際執行的下一動作（而非貪婪最大動作）來更新 Q 值，使其對探索策略更加敏感，在某些場景下更安全。

## 演算法命名由來

SARSA 的名字直接來自其更新的五元組：

$$(s_t, a_t, r_{t+1}, s_{t+1}, a_{t+1})$$

每個字母對應一個組成部分：
- **S**：當前狀態 $s_t$
- **A**：當前動作 $a_t$
- **R**：執行動作後得到的立即獎勵 $r_{t+1}$
- **S**：下一個狀態 $s_{t+1}$
- **A**：在 $s_{t+1}$ 實際選擇的下一個動作 $a_{t+1}$

相較之下，Q-Learning 只需要四元組 $(s, a, r, s')$，因為它的目標是 $\max_{a'} Q(s', a')$，不需要知道實際的 $a'$。

## SARSA 更新規則

SARSA 的更新公式如下：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

其中：
- $\alpha$ 為學習率（learning rate），控制單步更新的幅度
- $\gamma$ 為折扣因子（discount factor）
- $r$ 為立即獎勵
- $s'$ 為下一狀態
- $a'$ 為在 $s'$ 下根據當前行為策略（behavior policy）實際選擇的動作
- $\delta = r + \gamma Q(s', a') - Q(s, a)$ 為 TD 誤差（TD error）

完整的 SARSA 演算法偽代碼：

```
初始化 Q(s, a) 為任意值（通常為 0）
對於每個回合：
    初始化狀態 s
    使用 ε-greedy 選擇 a ← π(s)
    
    對每個時間步：
        執行動作 a，觀察 r, s'
        使用 ε-greedy 選擇 a' ← π(s')
        
        Q(s, a) ← Q(s, a) + α[r + γ Q(s', a') - Q(s, a)]
        
        s ← s', a ← a'
    直到 s 為終止狀態
```

## 在策略與離策略的根本差異

SARSA 和 Q-Learning 的核心差異在於目標計算方式：

### Q-Learning 的目標（離策略）

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$

Q-Learning 使用 $\max_{a'} Q(s', a')$，這對應於**貪婪策略**的 Q 值，完全忽略實際採用的策略。即使行為策略正在探索（隨機選擇動作），目標仍然是最佳可能的路徑。

### SARSA 的目標（在策略）

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

SARSA 使用 $Q(s', a')$，其中 $a'$ 是實際按照當前行為策略（包含探索）所選擇的動作。這意味著 SARSA 的 Q 值預期了探索的存在。

### 直覺理解

想像一個懸崖邊緣的任務（Cliff Walking）：
- Q-Learning 學到的策略會沿懸崖邊緣走（因為最佳路徑最短）
- SARSA 學到的策略會遠離懸崖邊緣（因為它意識到探索時可能掉下去）

這就是為什麼 SARSA 被稱為**保守的**，而 Q-Learning 被稱為**進取的**。

## 多步 SARSA（n-step SARSA）

SARSA(0) 只使用一步的獎勵資訊來更新 Q 值。n-step SARSA 將這個窗口擴展到 n 步，使用 n 步回報：

$$G_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \cdots + \gamma^{n-1} r_{t+n} + \gamma^n Q(s_{t+n}, a_{t+n})$$

相應的更新規則：

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ G_t^{(n)} - Q(s_t, a_t) \right]$$

### n 的選擇

| n | 方法 | 偏差 | 變異數 | 學習速度 |
|---|------|------|--------|---------|
| 0 | SARSA(0) | 高 | 低 | 慢（資訊傳播慢） |
| 1 | 1-step SARSA | 中等 | 低 | 中等 |
| 3-10 | n-step SARSA | 低 | 中等 | 快 |
| ∞ | Monte Carlo | 無 | 高 | 依賴完整回合 |

n 的選擇需要在偏差和變異數之間取得平衡。較小的 n（高偏差、低變異數）適合確定性環境，較大的 n（低偏差、高變異數）適合隨機性環境但需要更多樣本。

## 收斂性

### SARSA 的收斂條件

SARSA 被證明在以下條件下收斂到最優策略：

1. **有限狀態和動作空間的 MDP**
2. **策略 $\pi_t$ 是貪婪的極限（GLIE, Greedy in the Limit with Infinite Exploration）**：
   - 所有 $(s, a)$ 被无限次訪問
   - 學習率滿足 $\sum \alpha_t = \infty$ 且 $\sum \alpha_t^2 < \infty$
   - 探索率 $\epsilon_t \to 0$ 且 $\sum \epsilon_t = \infty$（如 $\epsilon_t = 1/t$）
3. **每個回合的 Q 值更新使用 $a' \sim \pi(\cdot|s')$**（在策略條件）

### 收斂速度比較

| 特性 | SARSA | Q-Learning |
|------|-------|------------|
| 收斂速度（初期） | 更快（保守、穩定） | 較慢（可能先學到偏差較大的值） |
| 最終性能 | 收斂到最優 $\epsilon$-soft 策略 | 收斂到最優確定性策略 |
| 探索敏感性 | 高（依賴探索策略） | 低（目標與探索分離） |
| 變異數 | 較低 | 較高（max 操作引入正偏差，稱為 maximization bias） |
| 安全風險 | 低（會避開探索時的危險） | 高（學習緊貼邊界的最短路徑） |

### Maximization Bias（最大化偏差）

Q-Learning 中的 $\max_{a'} Q(s', a')$ 操作引入了一個系統性的正偏差。即使 Q 值的估計是無偏的，取最大值後期望會偏高：

$$\mathbb{E} \left[ \max_{a'} Q(s', a') \right] \geq \max_{a'} \mathbb{E} \left[ Q(s', a') \right]$$

這類似於統計學中：即使每個隨機變數的期望為 0，它們的最大值的期望是大於 0 的。這個偏差在動作數量多或估計噪聲大時尤其嚴重。

SARSA 使用 $Q(s', a')$ 而非 $\max$，從根本上避免了這個問題。如果使用 Q-Learning 且發現學到的 Q 值系統性偏高，可以考慮改用 SARSA 或使用 Double Q-Learning（將動作選擇和價值估計分離到兩個獨立的 Q 函數）。

## 何時使用 SARSA vs Q-Learning

### 偏好 SARSA 的場景

- **安全關鍵**：如機器人控制、自動駕駛，真實環境中掉入懸崖不可接受
- **探索是必要的**：環境中探索本身就帶有風險或懲罰
- **在線學習**：策略邊學邊用，不能忍受離線過擬合的風險
- **高隨機性環境**：轉移機率不確定性很高時，SARSA 更穩定

### 偏好 Q-Learning 的場景

- **模擬環境**：可在低成本下大量模擬，探索的風險為 0
- **樣本效率優先**：需要從過往經驗中反覆學習（經驗回放 buffer）
- **追求理論最優**：確保最終策略是最大化 Q 值的確定性策略
- **應用深度學習函數逼近**：DQN 及其變體使用 Q-Learning 框架

## TD(λ) 與資格跡

SARSA 的單步版本（SARSA(0)）只使用一步的 TD 誤差來更新 Q 值。**TD(λ)** 演算法引入**資格跡（Eligibility Trace, E(s,a)）**的概念，將單步更新推廣到多步回溯：

### 資格跡的思想

單步 TD（TD(0)）的問題：資訊傳播速度慢，每步只能向後傳播一步的獎勵資訊。

以 FrozenLake 為例：如果智慧體在第 10 步才掉入洞中並得到懲罰，TD(0) 需要至少 10 個回合才能將這個懲罰回傳到第一步的決策。

資格跡解決方案：為每個 $(s,a)$ 維護一個**信用分數** $E(s,a)$，每次遇到好的或壞的事件，不只更新當前 $(s,a)$，也根據信用分數更新歷史上的所有 $(s,a)$。

### 更新規則

$$E(s,a) \leftarrow \gamma \lambda E(s,a) + \mathbf{1}(s_t=s, a_t=a)$$

$$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E(s,a)$$

其中：
- $\lambda \in [0, 1]$ 控制回溯深度
- $\gamma$ 為折扣因子（也決定資格跡的衰減速度）
- $\delta_t = r_{t+1} + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t)$ 為 TD 誤差
- $\mathbf{1}(\cdot)$ 為指示函數，只有當前的 $(s,a)$ 獲得 +1

### λ 的意義

```mermaid
graph LR
    subgraph TD(λ) 光譜
        L0[λ=0 → TD(0)] --> Lmid[λ∈(0,1) → 混合]
        Lmid --> L1[λ=1 → MC]
    end
    L0 -->|僅回傳一步| D0[高偏差, 低變異數]
    L1 -->|完整回合| D1[無偏差, 高變異數]
```

- **λ = 0**：退化成 SARSA(0)，只使用一步 TD 誤差。偏差最大，變異數最小。
- **λ = 1**：接近蒙地卡羅方法（Monte Carlo），使用完整回合回報。無偏差，但變異數最大。
- **λ 介於 0 和 1 之間**：在偏差和變異數之間取得平衡，通常表現最好。

### 累積資格跡 vs 替代資格跡

上述的 $E(s,a) \leftarrow \gamma \lambda E(s,a) + 1$ 是**累積跡（accumulating trace）**，每次訪問 $(s,a)$ 時直接加 1。這可能導致頻繁訪問的狀態-動作對獲得過高的 E 值。

另一種是**替代跡（replacing trace）**：

$$E(s,a) \leftarrow \gamma \lambda E(s,a) + (1 - \gamma \lambda) \cdot \mathbf{1}(s_t=s, a_t=a)$$

或更簡單的版本：

$$E(s,a) \leftarrow \gamma \lambda E(s,a)$$
$$E(s,a) \leftarrow 1 \quad \text{（若 s_t=s, a_t=a）} \quad \text{（直接設為 1，而非累加）}$$

替代跡在大多數任務中表現更穩定。

## 本專案中的實現

`world/examples/frozenlake_qtable.py` 實現了 SARSA、Q-Learning 和 TD(λ) 三種演算法的並列比較：

### SARSA 實現

```python
# SARSA：使用實際選擇的 a' 對應的 Q(s',a') 進行在策略更新
a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
Q[s, a] += alpha * (reward + gamma * Q[s1, a1] - Q[s, a])
```

關鍵點：$a_1$ 必須使用與當前相同的 ε-greedy 策略來選擇，這保證了在策略性質。

### TD(λ) 實現

```python
# TD(λ)：結合 TD 誤差與資格跡
a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
delta = reward + gamma * Q[s1, a1] - Q[s, a]  # TD 誤差

# 全局衰減資格跡，再對當前 (s,a) 增量
for s2 in range(env.observation_space.n):
    for a2 in range(env.action_space.n):
        E[s2, a2] *= gamma * lambda_
E[s,a] += 1

Q[s, a] += alpha * delta * E[s, a]
```

這裡的資格跡實現是累積跡（accumulating trace），每步對所有 $(s,a)$ 乘以 $\gamma \lambda$ 做全局衰減，再對當前的 $(s,a)$ 加 1。這在 16 狀態 × 4 動作的小規模 Q 表上是可行的，但面對大狀態空間時需要更高效的實現（如只追蹤非零的資格跡）。

### 三者的比較

在 FrozenLake-v1 環境中：
- **Q-Learning**：通常最快收斂到高成功率的策略，但不避開洞口的附近
- **SARSA**：收斂較慢，但學到的策略更穩健（會從洞口附近繞路走）
- **TD(λ)**：當 λ 取值適當時，學習速度比 TD(0) 快 2-5 倍，最終策略質量也更高

## Expected SARSA

Expected SARSA 是 SARSA 的一個變體，使用 Q 值的期望而非單一樣本：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \mathbb{E}_{a' \sim \pi} \left[ Q(s', a') \right] - Q(s, a) \right]$$

其中期望的計算方式為：

$$\mathbb{E}_{a' \sim \pi} \left[ Q(s', a') \right] = \sum_{a'} \pi(a'|s') Q(s', a')$$

Expected SARSA 的優點：
- **更低的變異數**：使用期望代替單一樣本，減少估計噪聲
- **更好的樣本效率**：每步更新的資訊量更大
- **橋接 SARSA 和 Q-Learning**：當 $\pi$ 是貪婪策略時退化成 Q-Learning；當 $\pi$ 是 ε-greedy 時是在策略學習

Expected SARSA 的缺點是需要知道 $\pi(a'|s')$ 的確切分布，這在 ε-greedy 策略下不成問題，但在連續動作空間或複雜策略分布下計算期望代價較高。

## SARSA 與深度學習的結合

傳統 SARSA 使用線性函數逼近或 Q 表，但現代深度 SARSA 使用類神經網路逼近 Q(s,a;w)：

$$\nabla_w L(w) = \mathbb{E} \left[ \left( r + \gamma Q(s', a'; w) - Q(s, a; w) \right) \nabla_w Q(s, a; w) \right]$$

與深度 Q-Learning（DQN）相比，深度 SARSA 的差異：
- **不需要 target network 的強制需求**（因為在策略性質使更新更穩定）
- **但樣本效率更低**（不能使用 experience replay 隨機採樣，因為需要 $a'$ 來自當前策略）
- 可以在 replay buffer 中儲存完整軌跡並按在策略重要性取樣（但實現複雜）

## 實務建議

1. **λ 的選擇**：對 FrozenLake 這類小規模任務，λ 在 0.2-0.4 效果較好；對大規模連續任務通常需要更小的 λ（0.1-0.3）。
2. **資格跡的重要性**：在獎勵稀疏的任務中，TD(λ) 的加速效果最明顯（獎勵資訊傳播更快）。
3. **與深度學習結合**：傳統的累積資格跡難以與神經網路函數逼近結合（非線性逼近器可能不穩定，需要 GT(λ) 或 Retrace 等改良版本）。
4. **SARSA vs Q-Learning 的選擇**：當環境成本高或安全重要時（如機器人訓練），SARSA 一直是合理的默認選擇。
5. **探索率衰減（epsilon decay）**：在 SARSA 中探索率影響最終策略，ε 不宜過早降為 0，否則策略可能卡在次優解。常用的調度策略是線性衰減、指數衰減或 $\epsilon = 1/t$。

---

**相關連結**：[Q-Learning.md](Q-Learning.md) | [Reinforcement-Learning.md](Reinforcement-Learning.md) | [FrozenLake-Environment.md](FrozenLake-Environment.md)
