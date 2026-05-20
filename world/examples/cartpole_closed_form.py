"""
CartPole Closed-Form Controller
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Like Gym, just set render_mode and call env.render() in loop.

Run:
    PYTHONPATH=. python world/examples/cartpole_closed_form.py
    PYTHONPATH=. python world/examples/cartpole_closed_form.py --render ansi
"""

import world

env = world.make("CartPole-v1", render_mode="human")
observation, _ = env.reset(seed=42)
steps = 0

try:
    for _ in range(2000):
        env.render()

        if observation[2] > 0:
            if observation[3] > 0.01:
                action = 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
            else:
                action = 0
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
        elif observation[2] < 0:
            if observation[3] < -0.01:
                action = 0
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1
            else:
                action = 1
                observation, _, terminated, truncated, _ = env.step(action)
                steps += 1

        if terminated or truncated:
            print(f"Episode ended: steps={steps}")
            steps = 0
            observation, _ = env.reset()
except KeyboardInterrupt:
    pass
finally:
    env.close()