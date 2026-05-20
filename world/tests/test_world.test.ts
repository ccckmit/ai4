import { Discrete, Box } from '../spaces';
import { make, registry } from '../utils/registry';
import { TimeLimitWrapper, RecordEpisodeWrapper } from '../wrappers';

describe('Spaces', () => {
  test('Discrete contains valid values', () => {
    const sp = new Discrete(4);
    for (let i = 0; i < 50; i++) {
      const s = sp.sample();
      expect(sp.contains(s)).toBe(true);
    }
    expect(sp.contains(0)).toBe(true);
    expect(sp.contains(3)).toBe(true);
    expect(sp.contains(4)).toBe(false);
    expect(sp.contains(-1)).toBe(false);
  });

  test('Box contains valid samples', () => {
    const sp = new Box(-1.0, 1.0, [3]);
    for (let i = 0; i < 50; i++) {
      const s = sp.sample();
      expect(sp.contains(s)).toBe(true);
    }
    expect(sp.contains([0, 0.5, -0.9])).toBe(true);
    expect(sp.contains([1.1, 0, 0])).toBe(false);
  });
});

describe('FrozenLake', () => {
  test('reset returns start state', () => {
    const env = make('FrozenLake-v1') as any;
    const result = env.reset({ seed: 0 });
    expect(result.observation).toBe(0);
    env.close();
  });

  test('step returns valid result', () => {
    const env = make('FrozenLake-v0') as any;
    env.reset({ seed: 1 });
    const step = env.step(2);
    expect(typeof step.observation).toBe('number');
    expect(typeof step.reward).toBe('number');
    expect(typeof step.terminated).toBe('boolean');
    expect(typeof step.truncated).toBe('boolean');
    env.close();
  });

  test('full episode runs', () => {
    const env = make('FrozenLake-v0') as any;
    env.reset({ seed: 42 });
    let done = false;
    let steps = 0;
    while (!done && steps < 200) {
      const result = env.step(env.actionSpace.sample() as number);
      done = result.done;
      steps++;
    }
    env.close();
    expect(steps).toBeGreaterThan(0);
  });
});

describe('CartPole', () => {
  test('reset returns 4-dim obs', () => {
    const env = make('CartPole-v1') as any;
    const result = env.reset({ seed: 0 });
    expect(result.observation.length).toBe(4);
    env.close();
  });

  test('step returns reward 1.0', () => {
    const env = make('CartPole-v1') as any;
    env.reset({ seed: 0 });
    const result = env.step(0);
    expect(result.observation.length).toBe(4);
    expect(result.reward).toBe(1.0);
    env.close();
  });

  test('invalid action raises', () => {
    const env = make('CartPole-v1') as any;
    env.reset();
    expect(() => env.step(5)).toThrow();
    env.close();
  });
});

describe('Wrappers', () => {
  test('TimeLimitWrapper truncates after max_steps', () => {
    const env = make('CartPole-v1') as any;
    const wrapped = new TimeLimitWrapper(env, 5);
    wrapped.reset({ seed: 0 });
    for (let i = 0; i < 4; i++) {
      const result = wrapped.step(0);
      expect(!result.truncated || result.terminated).toBe(true);
    }
    const result = wrapped.step(0);
    expect(result.truncated || result.terminated).toBe(true);
    env.close();
  });

  test('RecordEpisodeWrapper records episodes', () => {
    const env = make('FrozenLake-v0') as any;
    const wrapped = new RecordEpisodeWrapper(env);
    for (let ep = 0; ep < 5; ep++) {
      wrapped.reset({ seed: ep });
      let done = false;
      while (!done) {
        const result = wrapped.step(wrapped.actionSpace.sample() as number);
        done = result.done;
      }
    }
    expect(wrapped.episode_stats.length).toBe(5);
    expect('mean_reward' in wrapped.summary()).toBe(true);
    env.close();
  });
});

describe('Registry', () => {
  test('contains all built-in envs', () => {
    const r = registry();
    expect(r.has('FrozenLake-v0')).toBe(true);
    expect(r.has('FrozenLake-v1')).toBe(true);
    expect(r.has('FrozenLake8x8-v1')).toBe(true);
    expect(r.has('CartPole-v1')).toBe(true);
    expect(r.has('BipedalWalker-v3')).toBe(true);
  });

  test('make unknown raises', () => {
    expect(() => make('NoSuchEnv-v999' as any)).toThrow();
  });
});