import numpy as np
import world

env = world.make("FrozenLake-v1")
print('env=', env)
print('env.observation_space=', env.observation_space)
print('env.action_space=', env.action_space)
print('env.metadata=', env.metadata)

# 超參數設定
alpha = 0.8          # 學習速率
gamma = 0.95         # 折扣因子
num_episodes = 2000  # 訓練回合數

# 初始化 Q 表：16 個狀態 × 4 個動作
Q = np.zeros([env.observation_space.n, env.action_space.n])

# 選擇演算法（取消註解切換）
# method = "Q"
method = "SARSA"
# method = "TD_LAMBDA"
if method == "TD_LAMBDA":
    lambda_ = 0.2           # 資格跡衰減率

for i in range(num_episodes):
    s, info = env.reset()

    if method == "TD_LAMBDA":
        # 每個回合初始化資格跡矩陣
        E = np.zeros([env.observation_space.n, env.action_space.n])

    for j in range(99):
        # ε-greedy 選擇動作：加入隨機噪聲實現探索，噪聲隨回合遞減
        a = np.argmax(Q[s,:] + np.random.randn(1, env.action_space.n) * (1./(i+1)))
        s1, reward, terminated, truncated, info = env.step(a)

        if method == "Q":
            # Q-Learning：使用 max_{a'} Q(s',a') 進行離策略更新
            Q[s,a] += alpha * (reward + gamma * np.max(Q[s1,:]) - Q[s,a])

        elif method == "SARSA":
            # SARSA：使用實際選擇的 a' 對應的 Q(s',a') 進行基於策略更新
            a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            Q[s, a] += alpha * (reward + gamma * Q[s1, a1] - Q[s, a])

        elif method == "TD_LAMBDA":
            # TD(λ)：結合 TD 誤差與資格跡，實現更平滑的信用分配
            a1 = np.argmax(Q[s1, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
            delta = reward + gamma * Q[s1, a1] - Q[s, a]  # TD 誤差

            # 全局衰減資格跡，再對當前 (s,a) 增量
            for s2 in range(env.observation_space.n):
                for a2 in range(env.action_space.n):
                    E[s2, a2] *= gamma * lambda_
            E[s,a] += 1

            Q[s, a] += alpha * delta * E[s, a]

        else:
            raise Exception(f"無法處理的 method={method}")

        s = s1
        if terminated == True:
            break

print('Q=', Q)

if method == "TD_LAMBDA":
    print('E=', E)

print('完成迭代，展示學習成果 ...')

# 使用訓練好的 Q 表進行視覺化展示（永遠取最佳動作）
env = world.make('FrozenLake-v1')
s, info = env.reset()
for i in range(100):
    env.render()
    a = np.argmax(Q[s,:])
    s1, reward, terminated, truncated, info = env.step(a)
    print('-'*30)
    s = s1
    if terminated == True:
        break