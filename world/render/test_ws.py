"""
CartPole-v1 with browser-based rendering via WebSocket.

Usage:
    PYTHONPATH=. python world/render/test_ws.py

Opens browser at http://localhost:8080 showing real-time CartPole animation.
"""

import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from world.envs.cartpole import CartPoleEnv
from world.render.server import send_frame, start_server, stop_server


def main():
    start_server(8080)

    print("  Waiting for browser connection (4s)...")
    time.sleep(4)

    episodes = 10
    max_steps = 500

    print("=" * 50)
    print("  CartPole-v1  ·  Closed-Form Controller  (Python WS)")
    print("=" * 50)

    for ep in range(episodes):
        env = CartPoleEnv(max_steps=max_steps)
        obs, _ = env.reset(seed=ep * 100)
        steps = 0

        for _ in range(max_steps):
            x, x_dot, theta, theta_dot = obs

            if theta > 0:
                action = 1 if theta_dot > 0.01 else 0
            else:
                action = 0 if theta_dot < -0.01 else 1

            result = env.step(action)
            obs = result.observation
            steps += 1

            send_frame(
                x=float(obs[0]),
                theta=float(obs[2]),
                steps=steps,
                reward=result.reward,
                done=result.terminated or result.truncated,
            )

            time.sleep(0.033)

            if result.terminated or result.truncated:
                break

        print(f"  Episode {ep + 1}: {steps} steps")

    print("=" * 50)
    print("  Done! Press Ctrl+C to stop.")
    print("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()


if __name__ == "__main__":
    main()
