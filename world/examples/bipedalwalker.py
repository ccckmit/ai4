"""
BipedalWalker-v3 Training Example (ai4/nn)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Algorithm: SAC (Soft Actor-Critic) using ai4.nn framework

This example demonstrates:
1. Using world framework's BipedalWalker-v3 environment
2. Training with ai4.nn-based SAC agent (pure NumPy)

Run:
    PYTHONPATH=. python world/examples/bipedalwalker_ai4nn.py
"""

import argparse
import os
import random
from dataclasses import dataclass

import numpy as np

import world
from nn import Tensor, Module, Linear, Sequential, ReLU, Tanh, mse_loss, Adam, cat


# ------------------------------------------------------------------ #
#  Configuration                                                       #
# ------------------------------------------------------------------ #
@dataclass
class Config:
    env_id: str = "BipedalWalker-v3"
    seed: int = 42

    # SAC
    gamma: float = 0.99
    tau: float = 0.005
    tune_alpha: bool = True

    # Network
    hidden_dim: int = 128  # Smaller for CPU

    # Training
    total_steps: int = 50000
    batch_size: int = 64
    buffer_size: int = 50000
    learning_starts: int = 1000
    lr: float = 0.001
    updates_per_step: int = 1

    # Logging
    log_interval: int = 2000
    save_interval: int = 10000
    save_dir: str = "checkpoints_ai4"


LOG_STD_MIN, LOG_STD_MAX = -20, 2


# ------------------------------------------------------------------ #
#  Replay Buffer                                                        #
# ------------------------------------------------------------------ #
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        obs = Tensor(self.obs[indices], requires_grad=True)
        actions = Tensor(self.actions[indices], requires_grad=True)
        rewards = Tensor(self.rewards[indices], requires_grad=True)
        next_obs = Tensor(self.next_obs[indices], requires_grad=True)
        dones = Tensor(self.dones[indices], requires_grad=True)
        return obs, actions, rewards, next_obs, dones


# ------------------------------------------------------------------ #
#  Networks                                                             #
# ------------------------------------------------------------------ #
class Actor(Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_scale: np.ndarray):
        super().__init__()
        self.act_scale = Tensor(act_scale, requires_grad=False)
        
        self.shared = Sequential(
            Linear(obs_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
        )
        self.mu_head = Linear(hidden_dim, act_dim)
        self.log_std_head = Linear(hidden_dim, act_dim)

    def __call__(self, x):
        return self.forward(x)

    def forward(self, obs: Tensor) -> Tensor:
        h = self.shared(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = Tensor(np.exp(log_std.data), requires_grad=True)
        
        # Reparameterization: mu + std * epsilon
        epsilon = Tensor(np.random.randn(*mu.data.shape).astype(np.float32), requires_grad=False)
        x_t = mu + std * epsilon
        action = x_t.tanh() * self.act_scale
        return action

    def get_action(self, obs: Tensor, deterministic: bool = False) -> np.ndarray:
        h = self.shared(obs)
        mu = self.mu_head(h)
        if deterministic:
            action = mu.tanh() * self.act_scale
        else:
            log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
            std = Tensor(np.exp(log_std.data), requires_grad=True)
            epsilon = Tensor(np.random.randn(*mu.data.shape).astype(np.float32), requires_grad=False)
            action = (mu + std * epsilon).tanh() * self.act_scale
        return action.data


class Critic(Module):
    def __call__(self, obs, action):
        return self.forward(obs, action)

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = Sequential(
            Linear(obs_dim + act_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, hidden_dim),
            ReLU(),
            Linear(hidden_dim, 1),
        )

    def forward(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.net(cat([obs, action], axis=1))


# ------------------------------------------------------------------ #
#  SAC Agent                                                            #
# ------------------------------------------------------------------ #
class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_scale: np.ndarray,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
    ):
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(obs_dim, act_dim, hidden_dim, act_scale)
        self.critic1 = Critic(obs_dim, act_dim, hidden_dim)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim)

        # Target networks
        self.target_critic1 = Critic(obs_dim, act_dim, hidden_dim)
        self.target_critic2 = Critic(obs_dim, act_dim, hidden_dim)
        self._copy_weights(self.target_critic1, self.critic1)
        self._copy_weights(self.target_critic2, self.critic2)

        # Optimizers
        self.actor_opt = Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = Adam(
            self.critic1.parameters() + self.critic2.parameters(),
            lr=lr,
        )

        # Entropy temperature
        self.log_alpha = Tensor([0.0], requires_grad=True)
        self.alpha_opt = Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

    def _copy_weights(self, target, source):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data[:] = sp.data[:]

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = Tensor(obs.reshape(1, -1), requires_grad=False)
        action = self.actor.get_action(obs_t, deterministic)
        return action.squeeze()

    def update(self, batch) -> dict:
        obs, actions, rewards, next_obs, dones = batch

        # Update critics (next_obs doesn't need grad for target)
        next_obs_no_grad = Tensor(next_obs.data, requires_grad=False)
        next_actions = self.actor(next_obs_no_grad)
        target_q1 = self.target_critic1(next_obs, next_actions)
        target_q2 = self.target_critic2(next_obs, next_actions)
        target_q = target_q1 * 0.5 + target_q2 * 0.5
        
        # target = r + gamma * (1-done) * target_q
        one_minus_dones = Tensor(np.ones_like(dones.data) - dones.data, requires_grad=False)
        target = rewards + Tensor(np.ones_like(rewards.data) * self.gamma, requires_grad=False) * one_minus_dones * target_q

        # Q1, Q2 losses
        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = mse_loss(q1, target) + mse_loss(q2, target)
        
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor
        new_actions = self.actor(obs)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = (q1_new + q2_new) * 0.5

        alpha = Tensor([np.exp(self.log_alpha.data[0])], requires_grad=True)
        actor_loss = -q_new.sum() + alpha * new_actions.abs().sum()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update alpha
        alpha_loss = -self.log_alpha * (q_new.mean() - Tensor([self.target_entropy], requires_grad=False))
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update
        self._soft_update()

        return {
            "actor_loss": actor_loss.data[0] if hasattr(actor_loss.data, '__iter__') else float(actor_loss.data),
            "critic_loss": float(critic_loss.data),
            "alpha": np.exp(self.log_alpha.data[0]),
        }

    def _soft_update(self):
        for target, source in [
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data[:] = tp.data * (1.0 - self.tau) + sp.data * self.tau


# ------------------------------------------------------------------ #
#  Training                                                             #
# ------------------------------------------------------------------ #
def train(env: world.Env, config: Config):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = env.action_space.high.astype(np.float32)

    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim)
    agent = SACAgent(
        obs_dim, act_dim, act_scale,
        config.hidden_dim, config.lr,
        config.gamma, config.tau,
    )

    obs, _ = env.reset(seed=config.seed)
    episode_count = 0
    episode_reward = 0.0

    for step in range(config.total_steps):
        # Collect experience
        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs, deterministic=False)

        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        buffer.add(obs, action, reward, next_obs, float(done))

        obs = next_obs
        episode_reward += reward

        if done:
            episode_count += 1
            obs, _ = env.reset()
            episode_reward = 0.0

        # Update
        if step >= config.learning_starts:
            for _ in range(config.updates_per_step):
                batch = buffer.sample(config.batch_size)
                metrics = agent.update(batch)

        # Logging
        if (step + 1) % config.log_interval == 0:
            print(f"Step {step+1}/{config.total_steps} | Reward: {episode_reward:.2f} | Alpha: {metrics['alpha']:.3f}")

        # Save
        if (step + 1) % config.save_interval == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            path = os.path.join(config.save_dir, f"sac_step{step+1}.npz")
            # Save actor weights
            weights = {f"w{i}": p.data for i, p in enumerate(agent.actor.parameters())}
            np.savez(path, **weights)
            print(f"Saved: {path}")

    return agent


# ------------------------------------------------------------------ #
#  Main                                                                 #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="SAC BipedalWalker (ai4/nn)")
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(total_steps=args.steps, seed=args.seed)

    print(f"[Env] {config.env_id}")
    print(f"[Framework] ai4/nn (NumPy-based)")

    env = world.make(config.env_id)
    print(f"[Obs] {env.observation_space.shape}, [Act] {env.action_space.shape}")

    agent = train(env, config)
    print("Training complete!")


if __name__ == "__main__":
    main()