"""
CartPole-v1 VPG (Vanilla Policy Gradient) with ai4/nn
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
REINFORCE algorithm implementation using ai4.nn framework

Run:
    PYTHONPATH=. python world/examples/cartpole_vpg_ai4nn.py
"""

import itertools
import logging
import numpy as np

import world
from nn import Tensor, Module, Linear, Sequential, ReLU, Adam

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=__import__('sys').stdout,
    datefmt='%H:%M:%S'
)

env = world.make('CartPole-v1')


class PolicyNet(Module):
    def __init__(self, obs_dim: int, action_n: int):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, action_n),
        )

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        logits = self.net(x)
        return logits.softmax(axis=1)


class VPGAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.obs_dim = env.observation_space.shape[0]
        
        self.policy_net = PolicyNet(self.obs_dim, self.action_n)
        self.optimizer = Adam(self.policy_net.parameters(), lr=0.005)

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        state = Tensor(observation.reshape(1, -1), requires_grad=False)
        probs = self.policy_net(state)
        
        # Sample action from categorical distribution
        probs_np = probs.data.flatten()
        action = np.random.choice(self.action_n, p=probs_np)
        
        if self.mode == 'train':
            self.trajectory += [observation, reward, terminated, action]
        return action

    def close(self):
        if self.mode == 'train':
            self.learn()

    def learn(self):
        # Extract trajectory
        states = [self.trajectory[i] for i in range(0, len(self.trajectory), 4)]
        rewards = [self.trajectory[i] for i in range(1, len(self.trajectory), 4)]
        actions = [self.trajectory[i] for i in range(3, len(self.trajectory), 4)]
        
        # Compute discounted returns
        T = len(rewards)
        discounted_returns = np.zeros(T, dtype=np.float32)
        running = 0.0
        for t in reversed(range(T)):
            running = rewards[t] + self.gamma * running
            discounted_returns[t] = running
        
        discounted_returns = discounted_returns / (discounted_returns.std() + 1e-8)
        
        # Compute loss and gradients using numerical gradient
        params = self.policy_net.parameters()
        eps = 1e-5
        
        for p in params:
            grad = np.zeros_like(p.data)
            original = p.data.copy()
            
            for i in range(p.data.size):
                p.data.flat[i] += eps
                
                loss_plus = self._compute_loss(states, actions, discounted_returns)
                
                p.data.flat[i] = original.flat[i] - eps
                loss_minus = self._compute_loss(states, actions, discounted_returns)
                
                grad.flat[i] = (loss_plus - loss_minus) / (2 * eps)
                p.data.flat[i] = original.flat[i]
            
            p.grad = grad
        
        self.optimizer.step()

    def _compute_loss(self, states, actions, discounted_returns):
        T = min(len(states), 100)
        loss = 0.0
        states_arr = np.array(states[:T], dtype=np.float32)
        
        for t in range(T):
            state_t = Tensor(states_arr[t:t+1], requires_grad=False)
            probs_t = self.policy_net(state_t)
            log_prob_t = np.log(probs_t.data[0, actions[t]] + 1e-8)
            loss -= discounted_returns[t] * log_prob_t
        
        return loss / T


agent = VPGAgent(env)


def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, info = env.reset(seed=seed)
    reward, terminated, truncated = 0., False, False
    agent.reset(mode=mode)
    episode_reward, elapsed_steps = 0., 0
    
    while True:
        action = agent.step(observation, reward, terminated)
        if render:
            env.render(mode="ansi")
        if terminated or truncated:
            break
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        elapsed_steps += 1
    agent.close()
    return episode_reward, elapsed_steps


# Training
logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    episode_reward, elapsed_steps = play_episode(env, agent, seed=episode, mode='train')
    episode_rewards.append(episode_reward)
    if episode % 50 == 0:
        logging.info('train episode %d: reward = %.2f, steps = %d',
                     episode, episode_reward, elapsed_steps)
    if np.mean(episode_rewards[-20:]) > 199:
        break

logging.info(f"Training complete! Total episodes: {len(episode_rewards)}")
logging.info(f"Average reward (last 20): {np.mean(episode_rewards[-20:]):.2f}")

# Test
logging.info('==== test ====')
test_rewards = []
for episode in range(100):
    episode_reward, _ = play_episode(env, agent)
    test_rewards.append(episode_reward)

logging.info('average episode reward = %.2f ± %.2f',
             np.mean(test_rewards), np.std(test_rewards))