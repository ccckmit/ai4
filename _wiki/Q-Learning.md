# Q-Learning（Q學習）

Q-Learning 是強化學習（Reinforcement Learning）中一種經典的**離策略（off-policy）**時間差分（Temporal Difference, TD）控制演算法，由 Watkins 在 1989 年提出。它讓智慧體能夠在未知環境中學習最優動作價值函數（optimal action-value function），無需事先知道環境的轉移機制。

## 核心概念

### 動作價值函數 Q(s, a)

Q(s, a) 代表在狀態 s 執行動作 a 後，之後所有時間步驟的預期累積折扣回饋（discounted cumulative reward）：

$$Q(s, a) = \mathbb{E}\_\pi \left[ \sum\_{t=0}^{\infty} \gamma^t r_t \mid s_0=s, a_0=a \right]$$

其中：
- $\gamma \in [0, 1]$ 為折扣因子（discount factor），控制未來獎勵的重要性
- $\pi$ 為策略（policy），定義每個狀態下選擇各動作的機率分布

### Q-Learning 更新規則

每當智慧體執行一個互動 (s, a, r, s')，演算法透過以下規則更新 Q 值：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max\_{a'} Q(s', a') - Q(s, a) \right]$$

各符號意義：
- $\alpha$ 為學習率（learning rate），控制每次更新的步幅大小
- $r$ 為立即獎勵（immediate reward）
- $s'$ 為下一個狀態
- $\max\_{a'} Q(s', a')$ 為下一狀態所有可能動作中的最大 Q 值（此為離策略目標的關鍵）

這個更新公式的直覺含義：我們希望 Q(s,a) 逼近「立即獎勵 r 加上未來最佳可能的折扣獎勵」。差值 $\delta = r + \gamma \max\_{a'} Q(s', a') - Q(s,a)$ 就是 TD 誤差（TD error），反映了我們的估計與目標之間的距離。

## 離策略與在策略的區別

Q-Learning 之所以稱為「離策略」，是因為它使用的目標（target）與當前正在改善的策略無關：

- **目標策略（target policy）**：總是選擇能最大化 Q 值的動作，即 $\pi(s) = \arg\max_a Q(s,a)$
- **行為策略（behavior policy）**：用於生成動作的策略，通常採用 ε-greedy 進行探索

相對地，SARSA 是在策略（on-policy）演算法，其更新目標依賴實際執行的下一個動作：

$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma Q(s', a') - Q(s, a) \right]$$

其中 $a'$ 是實際選擇的下一動作，而非最大值。這導致 Q-Learning 更進取（aggressive）地追求最佳策略，而 SARSA 更保守、適合安全性關鍵的應用。

## ε-greedy 探索策略

在學習初期，智慧體對環境幾乎一無所知，需要積極探索。ε-greedy 是一種簡單有效的探索策略：

- 以 ε 機率：隨機選擇一個動作（探索）
- 以 1-ε 機率：選擇當前 Q 值最高的動作（利用）

ε 的設定是成敗關鍵之一：
- ε 太大：浪費過多時間在爛的動作上，收斂慢
- ε 太小：容易陷入局部最優，錯過最佳策略

實務上常採用 ε 衰減（epsilon decay）：一開始 ε=1.0，隨著訓練逐步衰減到如 0.05，讓智慧體在前期充分探索環境，後期則趨近貪婪策略。

## 收斂性條件

Q-Learning 在以下條件下被證明會收斂到最佳 Q 函數：

1. **環境是有限狀態與動作空間的馬可夫決策過程（MDP）**
2. **每個狀態-動作對被无限次地拜訪**
3. **學習率滿足 $\sum \alpha_t = \infty$ 且 $\sum \alpha_t^2 < \infty$**（如 $\alpha_t = 1/t$）
4. **環境是確定性的，或者是離散的平穩 MDP**

在連續空間或函數逼近（function approximation）場景中，Q-Learning 的收斂性不再有保證，這也是深度 Q 網路（DQN）需要額外技術（如經驗回放、目标网络）輔助的原因。

## Q 表（Q-Table）

在離散有限狀態空間中，Q(s,a) 可以用一張二維表儲存，稱為 Q 表。狀態數為 |S|、動作數為 |A| 時，Q 表大小為 |S| × |A|。每個格子儲存對應狀態-動作 pair 的 Q 值估計。

Q 表的優點是簡單直觀、收斂有理論保證；缺點是無法處理大狀態空間（想想圍棋的 $10^{170}$ 個狀態）。這正是深度強化學習（Deep RL）用神經網路逼近 Q 函數的動機。

## 演算法變體

### SARS

SARSA（State-Action-Reward-State-Action）是 Q-Learning 的在策略版本，更新時使用實際執行的下一動作而非最大 Q 值。這使得 SARSA 對探索策略更敏感，在某些場景下更安全。

### TD(λ) 與資格跡

TD(λ) 是一個統一的框架，涵蓋 TD(0)（只考慮一步）和 TD(∞)（相當於蒙特卡洛）。引入資格跡（eligibility trace）E(s,a) 來實現更平滑的信用分配：

$$E(s,a) \leftarrow \gamma \lambda E(s,a) + \mathbf{1}(s_t=s, a_t=a)$$

$$Q(s,a) \leftarrow Q(s,a) + \alpha \delta_t E(s,a)$$

其中 $\lambda \in [0,1]$ 控制回溯深度，λ=0 退化为 TD(0)，λ=1 则接近蒙特卡洛方法。

## 與深度學習的結合：DQN

傳統 Q-Learning 無法處理影像等高維輸入。Deep Q-Network（DQN，Mnih et al., 2013）用卷積神經網路（CNN）代替 Q 表，並引入兩項關鍵技術：

1. **經驗回放（Experience Replay）**：將互動儲存到重播緩衝區，隨機抽樣打破時序相關性
2. **目標網路（Target Network）**：固定目標 Q 值一段時間，穩定訓練

後續又有 Double DQN（解決 Q 值過估計）、Dueling DQN（分離狀態價值與動作優勢）等改進。

## 應用場景

Q-Learning 及其變體廣泛應用於：
- 遊戲 AI（Atari 遊戲、圍棋）
- 機器人控制（路徑規劃、動作選擇）
- 推薦系統
- 自動駕駛決策
- 工業程序控制

本專案中的 `world/examples/frozen_lake_example.py` 展示了 Q-Learning 如何在 FrozenLake 環境中學習導航策略，`world/examples/frozenlake_qtable.py` 則進一步比較了 Q-Learning、SARSA 和 TD(λ) 三種演算法的表現。

---

**下一篇**：[Reinforcement-Learning.md](Reinforcement-Learning.md) | [Transformer.md](Transformer.md)

**相關連結**：[Backpropagation.md](Backpropagation.md) | [Gradient-Descent.md](Gradient-Descent.md)