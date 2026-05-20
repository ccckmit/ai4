"""
BipedalWalker-v3 Training Example (PyTorch + world)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Algorithm: SAC (Soft Actor-Critic)

This example demonstrates:
1. Using world framework's BipedalWalker-v3 environment
2. Training with PyTorch-based SAC agent

Run:
    PYTHONPATH=. python world/examples/bipedalwalker_sac.py
"""

import argparse
import os
import random
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import world


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
    hidden_dim: int = 256

    # Training
    total_steps: int = 100000
    batch_size: int = 256
    buffer_size: int = 100000
    learning_starts: int = 1000
    lr: float = 3e-4
    updates_per_step: int = 1

    # Logging
    log_interval: int = 5000
    save_interval: int = 20000
    save_dir: str = "checkpoints"


# ------------------------------------------------------------------ #
#  Device                                                               #
# ------------------------------------------------------------------ #
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------ #
#  Replay Buffer                                                        #
# ------------------------------------------------------------------ #
class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device: torch.device):
        self.capacity = capacity
        self.device = device
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
        return (
            torch.FloatTensor(self.obs[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_obs[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device),
        )


# ------------------------------------------------------------------ #
#  Networks                                                             #
# ------------------------------------------------------------------ #
LOG_STD_MIN, LOG_STD_MAX = -20, 2


class Actor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, act_scale: torch.Tensor):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)
        self.register_buffer("act_scale", act_scale)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.shared(obs)
        mu = self.mu_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x_t = dist.rsample()
        action = torch.tanh(x_t)
        return action * self.act_scale

    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        h = self.shared(obs)
        mu = self.mu_head(h)
        if deterministic:
            return torch.tanh(mu) * self.act_scale
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp()
        x_t = torch.distributions.Normal(mu, std).rsample()
        return torch.tanh(x_t) * self.act_scale


class Critic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


# ------------------------------------------------------------------ #
#  SAC Agent                                                            #
# ------------------------------------------------------------------ #
class SACAgent:
    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        act_scale: torch.Tensor,
        hidden_dim: int,
        lr: float,
        gamma: float,
        tau: float,
        device: torch.device,
    ):
        self.gamma = gamma
        self.tau = tau
        self.device = device

        self.actor = Actor(obs_dim, act_dim, hidden_dim, act_scale).to(device)
        self.critic1 = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.critic2 = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic1 = Critic(obs_dim, act_dim, hidden_dim).to(device)
        self.target_critic2 = Critic(obs_dim, act_dim, hidden_dim).to(device)

        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()),
            lr=lr,
        )

        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=lr)
        self.target_entropy = -act_dim

    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action = self.actor.get_action(obs_t, deterministic)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch) -> dict:
        obs, actions, rewards, next_obs, dones = batch

        # Update critics
        with torch.no_grad():
            next_actions = self.actor(next_obs)
            target_q1 = self.target_critic1(next_obs, next_actions)
            target_q2 = self.target_critic2(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        q1 = self.critic1(obs, actions)
        q2 = self.critic2(obs, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Update actor
        new_actions = self.actor(obs)
        q1_new = self.critic1(obs, new_actions)
        q2_new = self.critic2(obs, new_actions)
        q_new = torch.min(q1_new, q2_new)

        alpha = self.log_alpha.exp()
        actor_loss = -q_new.mean() + alpha * (new_actions.abs().mean())

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Update alpha
        alpha_loss = -(self.log_alpha * (q_new.mean() - self.target_entropy).detach())

        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Soft update
        self._soft_update()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha": alpha.item(),
        }

    def _soft_update(self):
        for target, source in [
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ]:
            for target_param, param in zip(target.parameters(), source.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - self.tau) + param.data * self.tau
                )


# ------------------------------------------------------------------ #
#  Training                                                             #
# ------------------------------------------------------------------ #
def train(env: world.Env, config: Config, device: torch.device):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_scale = torch.FloatTensor(env.action_space.high).to(device)

    buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim, device)
    agent = SACAgent(
        obs_dim, act_dim, act_scale,
        config.hidden_dim, config.lr,
        config.gamma, config.tau, device,
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
            print(f"Step {step+1}/{config.total_steps} | Reward: {episode_reward:.2f}")

        # Save
        if (step + 1) % config.save_interval == 0:
            os.makedirs(config.save_dir, exist_ok=True)
            path = os.path.join(config.save_dir, f"sac_step{step+1}.pt")
            torch.save({"actor": agent.actor.state_dict()}, path)
            print(f"Saved: {path}")

    return agent


# ------------------------------------------------------------------ #
#  Main                                                                 #
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="SAC BipedalWalker Training")
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = Config(total_steps=args.steps, seed=args.seed)
    device = get_device()

    print(f"[Device] {device}")
    print(f"[Env] {config.env_id}")

    env = world.make(config.env_id)
    print(f"[Obs] {env.observation_space.shape}, [Act] {env.action_space.shape}")

    agent = train(env, config, device)
    print("Training complete!")


if __name__ == "__main__":
    main()