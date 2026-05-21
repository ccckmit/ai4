"""
CartPole-v1 Random Agent & Heuristic Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Demonstrates the CartPole environment with:
1. Random agent baseline
2. Heuristic (closed-form) balancing controller

Run:
    PYTHONPATH=. python world/examples/cartpole_example.py
"""

import world


def run_random_agent(env, episodes=5):
    """Run episodes with random actions."""
    print("\n  Random agent:")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 500:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"    Episode {ep+1}: reward={total_reward:.0f}, steps={steps}")


def run_heuristic_controller(env, episodes=3):
    """Simple heuristic: push opposite direction of pole lean."""
    print("\n  Heuristic controller:")
    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        done = False
        steps = 0
        while not done and steps < 500:
            angle = obs[2]
            angular_velocity = obs[3]
            if angle > 0:
                action = 0 if angular_velocity < -0.01 else 1
            else:
                action = 1 if angular_velocity > 0.01 else 0
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
        print(f"    Episode {ep+1}: reward={total_reward:.0f}, steps={steps}")


def main():
    print("=" * 50)
    print("  CartPole-v1 Example")
    print("=" * 50)
    print(f"\n  State space: position, velocity, angle, angular_velocity")
    print(f"  Action space: 0=left, 1=right")
    print(f"  Goal: keep pole upright for 500 steps")

    env = world.make("CartPole-v1")
    run_random_agent(env)
    run_heuristic_controller(env)
    env.close()

    print("\n" + "=" * 50)
    print("  Done!")
    print("=" * 50)


if __name__ == "__main__":
    main()
