"""
Pong Closed-Form Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Heuristic controller for Pong-v1: track the ball with the paddle.

Usage:
    PYTHONPATH=. python world/examples/pong_closed_form.py
"""

import sys

import world


def closed_form_action(obs):
    ball_y = obs[1]
    paddle_y = obs[4]
    if ball_y > paddle_y + 0.05:
        return 1
    elif ball_y < paddle_y - 0.05:
        return 0
    else:
        return 0


def main():
    env = world.make("Pong-v1")
    episodes = 10
    max_steps = 1000

    print("=" * 50)
    print("  Pong-v1  ·  Closed-Form (Heuristic) Controller")
    print("=" * 50)

    total_steps = 0
    total_hits = 0

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep * 100)
        steps = 0

        for _ in range(max_steps):
            action = closed_form_action(obs)
            result = env.step(action)
            obs = result.observation
            steps += 1

            if result.terminated or result.truncated:
                total_hits += result.info.get("hits", 0)
                break

        total_steps += steps
        print(f"  Episode {ep + 1:>2}: {steps:>4} steps")

    print("=" * 50)
    print(f"  Average: {(total_steps / episodes):.1f} steps")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    main()
