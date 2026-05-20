"""
examples/cartpole_example.py
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates CartPole-v1 with a simple PD controller + random agent comparison.

Run:  python cartpole_example.py
"""

import numpy as np
import world
from world.wrappers import RecordEpisodeWrapper
from world.utils import run_random_agent


# ─────────────────────────────────────────────────────────────────────────────
#  PD Controller  (simple hand-crafted policy)
# ─────────────────────────────────────────────────────────────────────────────

class PDController:
    """Push left/right based on pole angle and angular velocity."""

    def __init__(self, kp: float = 25.0, kd: float = 5.0):
        self.kp = kp
        self.kd = kd

    def act(self, obs: np.ndarray) -> int:
        """Return 0 (left) or 1 (right)."""
        _, _, theta, theta_dot = obs
        signal = self.kp * theta + self.kd * theta_dot
        return 1 if signal > 0 else 0


def run_pd_agent(episodes: int = 10, render_last: bool = True):
    env = world.make("CartPole-v1")
    recorder = RecordEpisodeWrapper(env)

    print("=" * 55)
    print("  world  ·  CartPole-v1  ·  PD Controller")
    print("=" * 55)

    controller = PDController()

    for ep in range(1, episodes + 1):
        obs, _ = recorder.reset(seed=ep)
        while True:
            action = controller.act(obs)
            result = recorder.step(action)
            obs = result.observation
            if result.done:
                break

    stats = recorder.summary()
    print(f"  Episodes      : {stats['episodes']}")
    print(f"  Mean reward   : {stats['mean_reward']:.1f}")
    print(f"  Max  reward   : {stats['max_reward']:.1f}")
    print(f"  Mean length   : {stats['mean_length']:.1f}")
    print("=" * 55)

    if render_last:
        print("\n  Rendering final episode with PD controller:\n")
        obs, _ = env.reset(seed=999)
        env.render()
        for _ in range(500):
            action = controller.act(obs)
            result = env.step(action)
            obs = result.observation
            print(f"  action={'→' if action else '←'}  x={result.info['x']:+.3f}  θ={result.info['theta_deg']:+.1f}°  reward={result.reward:.0f}")
            if result.done:
                status = "TERMINATED" if result.terminated else "TRUNCATED (max steps)"
                print(f"\n  Episode ended: {status} after {result.info['steps']} steps")
                break
        env.render()
        env.close()


def compare_random_vs_pd(episodes: int = 20):
    print("\n" + "=" * 55)
    print("  Comparison: Random agent  vs  PD controller")
    print("=" * 55)

    # Random
    print("\n  [Random Agent]")
    random_rewards = []
    env = world.make("CartPole-v1")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        while True:
            result = env.step(env.action_space.sample())
            total += result.reward
            obs = result.observation
            if result.done:
                break
        random_rewards.append(total)
    env.close()
    print(f"  Mean reward: {np.mean(random_rewards):.1f}  ±  {np.std(random_rewards):.1f}")

    # PD
    print("\n  [PD Controller]")
    pd_rewards = []
    controller = PDController()
    env = world.make("CartPole-v1")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total = 0.0
        while True:
            result = env.step(controller.act(obs))
            total += result.reward
            obs = result.observation
            if result.done:
                break
        pd_rewards.append(total)
    env.close()
    print(f"  Mean reward: {np.mean(pd_rewards):.1f}  ±  {np.std(pd_rewards):.1f}")
    print("=" * 55)


if __name__ == "__main__":
    run_pd_agent(episodes=10, render_last=True)
    compare_random_vs_pd(episodes=20)
