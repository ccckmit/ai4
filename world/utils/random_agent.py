"""world/utils/random_agent.py - Quick random-agent runner."""
from __future__ import annotations
from typing import Optional
from ..core import Env


def run_random_agent(
    env: Env,
    episodes: int = 5,
    render: bool = False,
    seed: Optional[int] = None,
) -> None:
    """Run a random agent for *episodes* episodes and print statistics.

    Parameters
    ----------
    env : Env
        Any world environment.
    episodes : int
        Number of episodes to run.
    render : bool
        If True, call ``env.render()`` at each step.
    seed : int, optional
        Seed passed to the first ``reset()``.
    """
    print(f"\n{'='*50}")
    print(f"  Random agent on {env}")
    print(f"  obs_space={env.observation_space}  act_space={env.action_space}")
    print(f"{'='*50}")

    total_rewards = []
    for ep in range(1, episodes + 1):
        ep_seed = seed + ep - 1 if seed is not None else None
        obs, info = env.reset(seed=ep_seed)
        total_reward = 0.0
        steps = 0

        while True:
            action = env.action_space.sample()
            result = env.step(action)
            total_reward += result.reward
            steps += 1

            if render:
                env.render()

            if result.done:
                status = "TERMINATED" if result.terminated else "TRUNCATED"
                print(f"  Episode {ep:3d} | {status:10s} | steps={steps:4d} | reward={total_reward:.1f}")
                total_rewards.append(total_reward)
                break

    avg = sum(total_rewards) / len(total_rewards)
    print(f"{'─'*50}")
    print(f"  Mean reward over {episodes} episodes: {avg:.3f}")
    print(f"{'='*50}\n")
    env.close()
