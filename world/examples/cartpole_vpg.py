"""
CartPole-v1 VPG (Vanilla Policy Gradient / REINFORCE) with PyTorch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Based on: https://zhiqingxiao.github.io/rl-book/en2024/code/CartPole-v0_VPG_torch.html

Run:
    PYTHONPATH=. python world/examples/cartpole_vpg.py
"""

import sys
import logging
import itertools

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributions as distributions

import world

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    stream=sys.stdout,
    datefmt='%H:%M:%S'
)

env = world.make('CartPole-v1')


class VPGAgent:
    def __init__(self, env):
        self.action_n = env.action_space.n
        self.gamma = 0.99

        self.policy_net = nn.Sequential(
            nn.Linear(env.observation_space.shape[0], 128),
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

    def get_action(self, observation):
        """Get action for inference (greedy)."""
        state_tensor = torch.as_tensor(observation, dtype=torch.float).unsqueeze(0)
        with torch.no_grad():
            prob_tensor = self.policy_net(state_tensor)
        return torch.argmax(prob_tensor, dim=1).item()

    def close(self):
        if self.mode == 'train':
            self.learn()

    def learn(self):
        states = torch.as_tensor(self.trajectory[0::4], dtype=torch.float)
        rewards = torch.as_tensor(self.trajectory[1::4], dtype=torch.float)
        actions = torch.as_tensor(self.trajectory[3::4], dtype=torch.long)

        T = states.shape[0]
        discounts = torch.pow(self.gamma, torch.arange(T, dtype=torch.float))

        discounted_returns = (discounts * rewards).flip(0).cumsum(0).flip(0)

        # Normalize returns (important!)
        discounted_returns = (discounted_returns - discounted_returns.mean()) / (discounted_returns.std() + 1e-8)

        all_probs = self.policy_net(states)
        log_probs = torch.log(torch.gather(all_probs, 1, actions.unsqueeze(1)).squeeze(1))
        loss = -(discounted_returns * log_probs).mean()

        self.optimizer.zero_grad()
        loss.backward()
        
        # Debug: check gradients
        # for name, param in self.policy_net.named_parameters():
        #     if param.grad is not None:
        #         print(f'{name}: grad norm = {param.grad.norm().item():.4f}')
        
        self.optimizer.step()


agent = VPGAgent(env)


def play_episode(env, agent, seed=None, mode=None, render=False):
    observation, info = env.reset(seed=seed)
    agent.reset(mode=mode)
    episode_reward = 0.0
    steps = 0
    done = False
    reward = 0.0

    while not done:
        action = agent.step(observation, reward, done)
        if render:
            env.render()
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        done = terminated or truncated

    agent.close()
    return episode_reward, steps


# Training
logging.info('==== train ====')
episode_rewards = []
for episode in itertools.count():
    reward, steps = play_episode(env, agent, seed=episode, mode='train')
    episode_rewards.append(reward)
    if episode % 100 == 0:
        logging.info('episode %d: reward = %.2f, steps = %d', episode, reward, steps)
    if len(episode_rewards) >= 20 and np.mean(episode_rewards[-20:]) > 199:
        break

logging.info(f'Training done! Total episodes: {len(episode_rewards)}')

# Test
logging.info('==== test ====')
test_rewards = []
for episode in range(100):
    reward, _ = play_episode(env, agent)
    test_rewards.append(reward)
    if episode % 20 == 0:
        logging.info('test episode %d: reward = %.2f', episode, reward)

logging.info('average test reward = %.2f ± %.2f', np.mean(test_rewards), np.std(test_rewards))

# Render demo
logging.info('==== render demo ====')
env_render = world.make('CartPole-v1', render_mode='human')
for ep in range(3):
    observation, _ = env_render.reset(seed=ep)
    done = False
    steps = 0
    while not done and steps < 500:
        env_render.render()
        action = agent.get_action(observation)
        observation, _, terminated, truncated, _ = env_render.step(action)
        done = terminated or truncated
        steps += 1
    logging.info(f'Demo episode {ep+1}: {steps} steps')
env_render.close()