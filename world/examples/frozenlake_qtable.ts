/**
 * examples/frozenlake_qtable.ts
 * Q-Learning, SARSA, and TD(λ) comparison for FrozenLake-v1
 */
import { make } from '../index';

const ALPHA = 0.8;
const GAMMA = 0.95;
const NUM_EPISODES = 2000;
const LAMBDA = 0.2;

const method = 'SARSA'; // 'Q', 'SARSA', or 'TD_LAMBDA'

function trainFrozenLake(): void {
  const env = make('FrozenLake-v1');
  const nStates = (env as any).observationSpace?.n ?? 16;
  const nActions = (env as any).actionSpace?.n ?? 4;

  const Q: number[][] = Array.from({ length: nStates }, () => new Array(nActions).fill(0));

  console.log('='.repeat(50));
  console.log(`  Training: ${method}`);
  console.log('='.repeat(50));

  for (let i = 0; i < NUM_EPISODES; i++) {
    const resetResult = env.reset();
    let s = resetResult.observation as number;

    let E: number[][] | null = null;
    if (method === 'TD_LAMBDA') {
      E = Array.from({ length: nStates }, () => new Array(nActions).fill(0));
    }

    for (let j = 0; j < 99; j++) {
      const noise = 1.0 / (i + 1);
      const action = argmaxWithNoise(Q[s], noise);

      const result = env.step(action);
      const s1 = result.observation as number;
      const reward = result.reward;
      const done = result.terminated || result.truncated;

      if (method === 'Q') {
        Q[s][action] += ALPHA * (reward + GAMMA * Math.max(...Q[s1]) - Q[s][action]);
      } else if (method === 'SARSA') {
        const a1 = argmaxWithNoise(Q[s1], noise);
        Q[s][action] += ALPHA * (reward + GAMMA * Q[s1][a1] - Q[s][action]);
      } else if (method === 'TD_LAMBDA') {
        const a1 = argmaxWithNoise(Q[s1], noise);
        const delta = reward + GAMMA * Q[s1][a1] - Q[s][action];

        for (let s2 = 0; s2 < nStates; s2++) {
          for (let a2 = 0; a2 < nActions; a2++) {
            E![s2][a2] *= GAMMA * LAMBDA;
          }
        }
        E![s][action] += 1;

        for (let s2 = 0; s2 < nStates; s2++) {
          for (let a2 = 0; a2 < nActions; a2++) {
            Q[s2][a2] += ALPHA * delta * E![s2][a2];
          }
        }
      }

      s = s1;
      if (done) break;
    }

    if ((i + 1) % 500 === 0) {
      const success = evaluate(env, Q);
      console.log(`  Episode ${i + 1}: success rate = ${(success * 100).toFixed(1)}%`);
    }
  }

  console.log('\n  Q table:');
  console.log(Q.map((row, i) => `  s${i}: [${row.map(v => v.toFixed(2)).join(', ')}]`).join('\n'));

  // Demo
  console.log('\n  Demonstrating learned policy ...');
  const demoEnv = make('FrozenLake-v1');
  const demoReset = demoEnv.reset();
  let s = demoReset.observation as number;
  for (let i = 0; i < 100; i++) {
    demoEnv.render();
    const a = Q[s].indexOf(Math.max(...Q[s]));
    const result = demoEnv.step(a);
    console.log('-'.repeat(30));
    s = result.observation as number;
    if (result.terminated) break;
  }
  demoEnv.close();
}

function argmaxWithNoise(Qs: number[], noise: number): number {
  const noisy = Qs.map(q => q + Math.random() * noise);
  let best = 0;
  for (let a = 1; a < noisy.length; a++) {
    if (noisy[a] > noisy[best]) best = a;
  }
  return best;
}

function evaluate(env: ReturnType<typeof make>, Q: number[][]): number {
  let wins = 0;
  for (let ep = 0; ep < 100; ep++) {
    const resetResult = env.reset({ seed: ep });
    let s = resetResult.observation as number;
    for (let step = 0; step < 99; step++) {
      const a = Q[s].indexOf(Math.max(...Q[s]));
      const result = env.step(a);
      s = result.observation as number;
      if (result.done) {
        if (result.reward > 0) wins++;
        break;
      }
    }
  }
  return wins / 100;
}

trainFrozenLake();