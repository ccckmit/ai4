"""
examples/frozen_lake_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
FrozenLake-v1 環境的 Q-Learning 範例，演示如何用離策略 TD 控制（Q-Learning）
讓智慧體在 4×4 冰湖地圖中學習避開洞穴（H）、抵達目標（G）的最優策略。

環境說明：
  FrozenLake-v1 為 4×4 格網世界，每格為以下四種之一：
    S (Start)  起始格，智慧體從此出發
    F (Frozen) 安全冰面，可通行
    H (Hole)   洞穴，踩到該格回合失敗（reward=0）
    G (Goal)   目標格，抵達後回合成功（reward=1）
  另外存在 slip 變化（v1 版本）：智慧體選擇動作後，有機率滑到相鄰格子，
  使問題具有隨機性挑戰性。

狀態與動作：
  狀態空間：16 個離散狀態（每格一個索引 0~15）
  動作空間：4 個動作（0=左、1=下、2=右、3=上）

Q-Learning 更新規則（離策略 TD 控制）：
  Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]
  其中：
    α  為學習率（控制更新的步幅）
    γ  為折扣因子（未來 reward 的重要性）
    max_a' Q(s',a') 為下一狀態所有動作 Q 值的最大值（離策略目標）

ε-greedy 探索策略：
  以 ε 機率隨機選擇動作（探索），以 (1-ε) 機率選擇當前最優動作（利用）。
  ε 隨訓練逐漸衰減（EPSILON_DECAY），前期鼓勵探索，後期趨近貪婪策略。

執行方式（需在專案根目錄）：
  PYTHONPATH=. uv run python world/examples/frozen_lake_example.py
"""

import numpy as np
import world
from world.wrappers import RecordEpisodeWrapper


# ─────────────────────────────────────────────
#  超參數（Hyper-parameters）
# ─────────────────────────────────────────────
EPISODES       = 5_000   # 訓練總回合數
MAX_STEPS      = 200     # 每回合最大步數（避免無限迴圈）
ALPHA          = 0.8     # 學習率（learning rate）：更新步幅，越大收斂越快但可能震盪
GAMMA          = 0.95    # 折扣因子（discount factor）：未來 reward 的衰減權重
               #   γ 接近 1 表示智慧體更重視長遠 reward，接近 0 則只重視立即 reward
EPSILON_START  = 1.0     # 初始 ε 值：訓練一開始 100% 隨機探索
EPSILON_END    = 0.05    # ε 衰減下限：訓練後期最少保留 5% 隨機探索，避免完全貪婪
EPSILON_DECAY  = 0.999   # ε 衰減率：每回合後 ε ← max(EPSILON_END, ε * DECAY)
               #   衰減後 ε ≈ 0.05（收斂前期），足夠探索又偏向貪婪
EVAL_EPISODES  = 200     # 訓練後評估回合數，用於計算勝率
SEED           = 0      # 隨機種子，確保可重現性


def epsilon_greedy(Q, state, epsilon, n_actions, rng):
    if rng.random() < epsilon:
        return rng.integers(n_actions)
    return int(np.argmax(Q[state]))


def train():
    env = world.make("FrozenLake-v1")           # slippery 4×4
    recorder = RecordEpisodeWrapper(env)

    n_states  = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions))

    rng     = np.random.default_rng(SEED)
    epsilon = EPSILON_START

    print("=" * 55)
    print("  world  ·  FrozenLake-v1  ·  Q-Learning")
    print("=" * 55)

    for ep in range(1, EPISODES + 1):
        obs, _ = recorder.reset(seed=int(rng.integers(1_000_000)))
        total_reward = 0.0

        for _ in range(MAX_STEPS):
            action = epsilon_greedy(Q, obs, epsilon, n_actions, rng)
            result = recorder.step(action)
            next_obs, reward, terminated, truncated, _ = result

            # Q update
            best_next = np.max(Q[next_obs])
            Q[obs, action] += ALPHA * (reward + GAMMA * best_next - Q[obs, action])

            obs = next_obs
            total_reward += reward
            if result.done:
                break

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        if ep % 500 == 0:
            stats = recorder.summary()
            last_n = recorder.episode_stats[-500:]
            win_rate = sum(1 for e in last_n if e["reward"] > 0) / len(last_n)
            print(f"  Episode {ep:5d} | ε={epsilon:.3f} | win_rate(500)={win_rate:.2%}")

    print("\n  Training complete!")
    print(f"  Total episodes recorded: {len(recorder.episode_stats)}")

    # ── Evaluation (greedy) ──────────────────────────────────────────
    print(f"\n  Evaluating greedy policy over {EVAL_EPISODES} episodes …")
    wins = 0
    eval_env = world.make("FrozenLake-v1")
    for ep in range(EVAL_EPISODES):
        obs, _ = eval_env.reset(seed=ep)
        for _ in range(MAX_STEPS):
            action = int(np.argmax(Q[obs]))
            result = eval_env.step(action)
            obs = result.observation
            if result.done:
                if result.reward > 0:
                    wins += 1
                break
    eval_env.close()
    print(f"  Win rate: {wins}/{EVAL_EPISODES} = {wins/EVAL_EPISODES:.1%}")
    print("=" * 55)

    # ── One rendered episode ─────────────────────────────────────────
    print("\n  Rendering one greedy episode:\n")
    demo_env = world.make("FrozenLake-v0")   # deterministic for clarity
    obs, _ = demo_env.reset(seed=SEED)
    demo_env.render()
    for step in range(MAX_STEPS):
        action = int(np.argmax(Q[obs]))
        result = demo_env.step(action)
        print(f"\n  → action={action}  reward={result.reward}  done={result.done}")
        demo_env.render()
        obs = result.observation
        if result.done:
            break
    demo_env.close()


if __name__ == "__main__":
    train()
