"""
CartPole-v1 VPG (Vanilla Policy Gradient) with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
REINFORCE algorithm implementation for CartPole-v1

Run:
    PYTHONPATH=. python world/examples/cartpole_vpg.py
"""

import itertools
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import world

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=__import__('sys').stdout,
    datefmt='%H:%M:%S'
)

env = world.make('CartPole-v1')


class VPGAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99
        self.obs_dim = env.observation_space.shape[0]
        
        # Policy network: input state, output action probabilities (Softmax)
        self.policy_net = nn.Sequential(
            nn.Linear(self.obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_n),
            nn.Softmax(dim=1)
        )
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.005)

    def reset(self, mode=None):
        self.mode = mode
        if self.mode == 'train':
            self.trajectory = []

    def step(self, observation, reward, terminated):
        state_tensor = torch.as_tensor(observation, dtype=torch.float).unsqueeze(0)
        prob_tensor = self.policy_net(state_tensor)
        action_tensor = distributions.Categorical(prob_tensor).sample()
        action = action_tensor.item()
        if self.mode == 'train':
            self.trajectory += [observation, reward, terminated, action]
        return action

    def close(self):
        if self.mode == 'train':
            self.learn()

    def learn(self):
        state_tensor = torch.as_tensor(self.trajectory[0::4], dtype=torch.float)
        reward_tensor = torch.as_tensor(self.trajectory[1::4], dtype=torch.float)
        action_tensor = torch.as_tensor(self.trajectory[3::4], dtype=torch.long)
        
        # Discount factor: [γ^0, γ^1, ..., γ^{T-1}]
        arange_tensor = torch.arange(state_tensor.shape[0], dtype=torch.float)
        discount_tensor = self.gamma ** arange_tensor
        
        # Discounted return: G_t = Σ_{k=t}^{T} γ^{k-t} r_k
        discounted_reward_tensor = discount_tensor * reward_tensor
        discounted_return_tensor = discounted_reward_tensor.flip(0).cumsum(0).flip(0)
        
        # Loss: -G_t log π(a_t|s_t)
        all_pi_tensor = self.policy_net(state_tensor)
        pi_tensor = torch.gather(all_pi_tensor, 1, action_tensor.unsqueeze(1)).squeeze(1)
        log_pi_tensor = torch.log(torch.clamp(pi_tensor, 1e-6, 1.))
        loss_tensor = -(discounted_return_tensor * log_pi_tensor).mean()
        
        self.optimizer.zero_grad()
        loss_tensor.backward()
        self.optimizer.step()


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