"""tests/test_world.py - Basic smoke tests."""
import numpy as np
import world
from world.spaces import Discrete, Box
from world.wrappers import TimeLimitWrapper, RecordEpisodeWrapper


# ─────────────────────────────────────────────────────────────────────────────
#  Space tests
# ─────────────────────────────────────────────────────────────────────────────

def test_discrete_space():
    sp = Discrete(4, seed=0)
    for _ in range(50):
        s = sp.sample()
        assert sp.contains(s), f"{s} not in {sp}"
    assert sp.contains(0)
    assert sp.contains(3)
    assert not sp.contains(4)
    assert not sp.contains(-1)
    print("  [PASS] Discrete space")


def test_box_space():
    sp = Box(-1.0, 1.0, shape=(3,), seed=0)
    for _ in range(50):
        s = sp.sample()
        assert sp.contains(s), f"{s} not in {sp}"
    assert sp.contains(np.array([0.0, 0.5, -0.9], dtype=np.float32))
    assert not sp.contains(np.array([1.1, 0.0, 0.0], dtype=np.float32))
    print("  [PASS] Box space")


# ─────────────────────────────────────────────────────────────────────────────
#  FrozenLake tests
# ─────────────────────────────────────────────────────────────────────────────

def test_frozen_lake_reset():
    env = world.make("FrozenLake-v1")
    obs, info = env.reset(seed=0)
    assert obs == 0, f"Expected start state 0, got {obs}"
    assert "pos" in info
    print("  [PASS] FrozenLake reset")


def test_frozen_lake_step():
    env = world.make("FrozenLake-v0")   # deterministic
    obs, _ = env.reset(seed=1)
    result = env.step(2)   # RIGHT
    assert isinstance(result.observation, (int, np.integer))
    assert isinstance(result.reward, float)
    assert isinstance(result.terminated, bool)
    assert isinstance(result.truncated, bool)
    print("  [PASS] FrozenLake step")


def test_frozen_lake_unpack():
    env = world.make("FrozenLake-v0")
    env.reset()
    obs, reward, terminated, truncated, info = env.step(1)
    assert isinstance(obs, (int, np.integer))
    print("  [PASS] FrozenLake tuple unpack")


def test_frozen_lake_full_episode():
    env = world.make("FrozenLake-v0")
    obs, _ = env.reset(seed=42)
    done = False
    steps = 0
    while not done and steps < 200:
        result = env.step(env.action_space.sample())
        obs = result.observation
        done = result.done
        steps += 1
    env.close()
    print(f"  [PASS] FrozenLake full episode ({steps} steps)")


# ─────────────────────────────────────────────────────────────────────────────
#  CartPole tests
# ─────────────────────────────────────────────────────────────────────────────

def test_cartpole_reset():
    env = world.make("CartPole-v1")
    obs, info = env.reset(seed=0)
    assert obs.shape == (4,), f"Expected shape (4,), got {obs.shape}"
    assert env.observation_space.contains(obs)
    print("  [PASS] CartPole reset")


def test_cartpole_step():
    env = world.make("CartPole-v1")
    env.reset(seed=0)
    result = env.step(0)
    assert result.observation.shape == (4,)
    assert result.reward == 1.0
    print("  [PASS] CartPole step")


def test_cartpole_full_episode():
    env = world.make("CartPole-v1")
    obs, _ = env.reset(seed=7)
    total_reward = 0.0
    done = False
    while not done:
        result = env.step(env.action_space.sample())
        total_reward += result.reward
        done = result.done
    env.close()
    assert total_reward >= 1.0
    print(f"  [PASS] CartPole full episode (reward={total_reward:.0f})")


def test_cartpole_invalid_action():
    env = world.make("CartPole-v1")
    env.reset()
    try:
        env.step(5)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("  [PASS] CartPole invalid action raises ValueError")


# ─────────────────────────────────────────────────────────────────────────────
#  Wrapper tests
# ─────────────────────────────────────────────────────────────────────────────

def test_time_limit_wrapper():
    env = TimeLimitWrapper(world.make("CartPole-v1"), max_steps=5)
    env.reset(seed=0)
    for i in range(4):
        result = env.step(0)
        assert not result.truncated or result.terminated
    result = env.step(0)
    assert result.truncated or result.terminated
    env.close()
    print("  [PASS] TimeLimitWrapper")


def test_record_episode_wrapper():
    env = RecordEpisodeWrapper(world.make("FrozenLake-v0"))
    for ep in range(5):
        env.reset(seed=ep)
        done = False
        while not done:
            result = env.step(env.action_space.sample())
            done = result.done
    assert len(env.episode_stats) == 5
    s = env.summary()
    assert "mean_reward" in s
    env.close()
    print(f"  [PASS] RecordEpisodeWrapper (mean_reward={s['mean_reward']:.2f})")


# ─────────────────────────────────────────────────────────────────────────────
#  Registry tests
# ─────────────────────────────────────────────────────────────────────────────

def test_registry():
    r = world.registry()
    for key in ["FrozenLake-v0", "FrozenLake-v1", "FrozenLake8x8-v1", "CartPole-v1"]:
        assert key in r, f"{key} not in registry"
    print("  [PASS] Registry contains all built-in envs")


def test_make_unknown():
    try:
        world.make("NoSuchEnv-v999")
        assert False
    except KeyError:
        pass
    print("  [PASS] make() raises KeyError for unknown env")


# ─────────────────────────────────────────────────────────────────────────────
#  Run all
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        test_discrete_space,
        test_box_space,
        test_frozen_lake_reset,
        test_frozen_lake_step,
        test_frozen_lake_unpack,
        test_frozen_lake_full_episode,
        test_cartpole_reset,
        test_cartpole_step,
        test_cartpole_full_episode,
        test_cartpole_invalid_action,
        test_time_limit_wrapper,
        test_record_episode_wrapper,
        test_registry,
        test_make_unknown,
    ]
    print(f"\n{'='*50}")
    print(f"  world test suite  ({len(tests)} tests)")
    print(f"{'='*50}")
    passed = 0
    for t in tests:
        try:
            t()
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {t.__name__}: {e}")
    print(f"{'─'*50}")
    print(f"  {passed}/{len(tests)} passed")
    print(f"{'='*50}\n")
