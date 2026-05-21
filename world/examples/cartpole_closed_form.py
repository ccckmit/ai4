"""
CartPole Closed-Form Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Heuristic PD-like controller for CartPole-v1.

Usage:
    PYTHONPATH=. python world/examples/cartpole_closed_form.py          # console only
    PYTHONPATH=. python world/examples/cartpole_closed_form.py --render # browser WebSocket
"""

import sys
import time

import world


def main():
    env = world.make("CartPole-v1")
    episodes = 10
    max_steps = 500
    render = "--render" in sys.argv

    print("=" * 50)
    print("  CartPole-v1  ·  Closed-Form (Heuristic) Controller")
    if render:
        print("  Render mode: browser (http://localhost:8080)")
    print("=" * 50)

    if render:
        print("  Waiting for browser connection (3s)...")
        time.sleep(3)

    total_steps = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep * 100)
        steps = 0

        for _ in range(max_steps):
            _, _, theta, theta_dot = obs
            action = 1 if theta > 0 and theta_dot > 0.01 else \
                     0 if theta > 0 else \
                     0 if theta < 0 and theta_dot < -0.01 else 1
            result = env.step(action)
            obs = result.observation
            steps += 1

            if render:
                env.render("human")
                time.sleep(0.033)

            if result.terminated or result.truncated:
                break

        total_steps += steps
        print(f"  Episode {ep + 1:>2}: {steps:>3} steps")

    print("=" * 50)
    print(f"  Average: {(total_steps / episodes):.1f} steps")
    print("=" * 50)

    env.close()

    if render:
        print("  Press Enter to stop server...")
        input()


if __name__ == "__main__":
    main()
