/**
 * tests/test_world.ts - Basic smoke tests for world package.
 */
import { Discrete, Box } from '../spaces';
import { Env, StepResult } from '../core';
import { TimeLimitWrapper, RecordEpisodeWrapper } from '../wrappers';
import { make, registry } from '../utils/registry';

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(`Assertion failed: ${msg}`);
}

function assertAlmostEq(a: number, b: number, tol = 1e-5) {
  if (Math.abs(a - b) > tol) throw new Error(`Expected ${a} ≈ ${b}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Space tests
// ─────────────────────────────────────────────────────────────────────────────

function testDiscreteSpace() {
  const sp = new Discrete(4);
  for (let i = 0; i < 50; i++) {
    const s = sp.sample();
    assert(sp.contains(s), `${s} not in Discrete(4)`);
  }
  assert(sp.contains(0), 'should contain 0');
  assert(sp.contains(3), 'should contain 3');
  assert(!sp.contains(4), 'should not contain 4');
  assert(!sp.contains(-1), 'should not contain -1');
  console.log('  [PASS] Discrete space');
}

function testBoxSpace() {
  const sp = new Box(-1.0, 1.0, [3]);
  for (let i = 0; i < 50; i++) {
    const s = sp.sample();
    assert(sp.contains(s), `${s} not in Box(-1,1,shape=3)`);
  }
  const inside = [0.0, 0.5, -0.9];
  assert(sp.contains(inside), 'should contain [0, 0.5, -0.9]');
  const outside = [1.1, 0.0, 0.0];
  assert(!sp.contains(outside), 'should not contain [1.1, 0, 0]');
  console.log('  [PASS] Box space');
}

// ─────────────────────────────────────────────────────────────────────────────
// FrozenLake tests
// ─────────────────────────────────────────────────────────────────────────────

function testFrozenLakeReset() {
  const env = make('FrozenLake-v1') as Env<number, number>;
  const result = env.reset({ seed: 0 });
  assert(result.observation === 0, `Expected start state 0, got ${result.observation}`);
  assert('pos' in result.info, 'info should have pos');
  console.log('  [PASS] FrozenLake reset');
}

function testFrozenLakeStep() {
  const env = make('FrozenLake-v0') as Env<number, number>;
  const result = env.reset({ seed: 1 });
  const stepResult = env.step(2); // RIGHT
  assert(typeof stepResult.observation === 'number', 'observation should be number');
  assert(typeof stepResult.reward === 'number', 'reward should be number');
  assert(typeof stepResult.terminated === 'boolean', 'terminated should be boolean');
  assert(typeof stepResult.truncated === 'boolean', 'truncated should be boolean');
  console.log('  [PASS] FrozenLake step');
}

function testFrozenLakeUnpack() {
  const env = make('FrozenLake-v0') as Env<number, number>;
  env.reset();
  const result = env.step(1);
  assert(typeof result.observation === 'number', 'observation should be number');
  console.log('  [PASS] FrozenLake tuple unpack');
}

function testFrozenLakeFullEpisode() {
  const env = make('FrozenLake-v0') as Env<number, number>;
  env.reset({ seed: 42 });
  let done = false;
  let steps = 0;
  const maxSteps = 200;
  while (!done && steps < maxSteps) {
    const result = env.step(env.actionSpace.sample() as number);
    done = result.done;
    steps++;
  }
  env.close();
  console.log(`  [PASS] FrozenLake full episode (${steps} steps)`);
}

// ─────────────────────────────────────────────────────────────────────────────
// CartPole tests
// ─────────────────────────────────────────────────────────────────────────────

function testCartPoleReset() {
  const env = make('CartPole-v1') as Env<number[], number>;
  const result = env.reset({ seed: 0 });
  assert(result.observation.length === 4, `Expected shape (4,), got (${result.observation.length},)`);
  console.log('  [PASS] CartPole reset');
}

function testCartPoleStep() {
  const env = make('CartPole-v1') as Env<number[], number>;
  env.reset({ seed: 0 });
  const result = env.step(0);
  assert(result.observation.length === 4, `Expected shape (4,), got (${result.observation.length},)`);
  assertAlmostEq(result.reward, 1.0);
  console.log('  [PASS] CartPole step');
}

function testCartPoleFullEpisode() {
  const env = make('CartPole-v1') as Env<number[], number>;
  env.reset({ seed: 7 });
  let totalReward = 0.0;
  let done = false;
  while (!done) {
    const result = env.step(env.actionSpace.sample() as number);
    totalReward += result.reward;
    done = result.done;
  }
  env.close();
  assert(totalReward >= 1.0, `Expected total_reward >= 1, got ${totalReward}`);
  console.log(`  [PASS] CartPole full episode (reward=${totalReward.toFixed(0)})`);
}

function testCartPoleInvalidAction() {
  const env = make('CartPole-v1') as Env<number[], number>;
  env.reset();
  try {
    env.step(5);
    assert(false, 'Should have raised Error');
  } catch (e) {
    // Expected
  }
  console.log('  [PASS] CartPole invalid action raises Error');
}

// ─────────────────────────────────────────────────────────────────────────────
// Wrapper tests
// ─────────────────────────────────────────────────────────────────────────────

function testTimeLimitWrapper() {
  const env = make('CartPole-v1') as Env<number[], number>;
  const wrapped = new TimeLimitWrapper(env, 5);
  wrapped.reset({ seed: 0 });
  for (let i = 0; i < 4; i++) {
    const result = wrapped.step(0);
    assert(!result.truncated || result.terminated, `step ${i} should not truncate`);
  }
  const result = wrapped.step(0);
  assert(result.truncated || result.terminated, 'step 5 should truncate or terminate');
  env.close();
  console.log('  [PASS] TimeLimitWrapper');
}

function testRecordEpisodeWrapper() {
  const env = make('FrozenLake-v0') as Env<number, number>;
  const wrapped = new RecordEpisodeWrapper(env);
  for (let ep = 0; ep < 5; ep++) {
    wrapped.reset({ seed: ep });
    let done = false;
    while (!done) {
      const result = wrapped.step(wrapped.actionSpace.sample() as number);
      done = result.done;
    }
  }
  assert(wrapped.episode_stats.length === 5, `Expected 5 episodes, got ${wrapped.episode_stats.length}`);
  const s = wrapped.summary();
  assert('mean_reward' in s, 'summary should have mean_reward');
  env.close();
  console.log(`  [PASS] RecordEpisodeWrapper (mean_reward=${s.mean_reward.toFixed(2)})`);
}

// ─────────────────────────────────────────────────────────────────────────────
// Registry tests
// ─────────────────────────────────────────────────────────────────────────────

function testRegistry() {
  const r = registry();
  for (const key of ['FrozenLake-v0', 'FrozenLake-v1', 'FrozenLake8x8-v1', 'CartPole-v1']) {
    assert(key in r, `${key} not in registry`);
  }
  console.log('  [PASS] Registry contains all built-in envs');
}

function testMakeUnknown() {
  try {
    make('NoSuchEnv-v999');
    assert(false, 'Should have thrown');
  } catch (e) {
    // Expected
  }
  console.log('  [PASS] make() raises Error for unknown env');
}

// ─────────────────────────────────────────────────────────────────────────────
// Run all
// ─────────────────────────────────────────────────────────────────────────────

const tests = [
  testDiscreteSpace,
  testBoxSpace,
  testFrozenLakeReset,
  testFrozenLakeStep,
  testFrozenLakeUnpack,
  testFrozenLakeFullEpisode,
  testCartPoleReset,
  testCartPoleStep,
  testCartPoleFullEpisode,
  testCartPoleInvalidAction,
  testTimeLimitWrapper,
  testRecordEpisodeWrapper,
  testRegistry,
  testMakeUnknown,
];

console.log('\n' + '='.repeat(50));
console.log(`  world test suite  (${tests.length} tests)`);
console.log('='.repeat(50));

let passed = 0;
for (const t of tests) {
  try {
    t();
    passed++;
  } catch (e: any) {
    console.log(`  [FAIL] ${t.name}: ${e.message}`);
  }
}
console.log('-'.repeat(50));
console.log(`  ${passed}/${tests.length} passed`);
console.log('='.repeat(50) + '\n');