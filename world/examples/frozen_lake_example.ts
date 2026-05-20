/**
 * examples/frozen_lake_example.ts
 * FrozenLake-v1 Q-Learning example.
 * Train an agent to navigate 4x4 FrozenLake using off-policy TD control.
 */
import { make } from '../index';
import { RecordEpisodeWrapper } from '../wrappers/record_episode';

const EPISODES = 5000;
const MAX_STEPS = 200;
const ALPHA = 0.8;
const GAMMA = 0.95;
const EPSILON_START = 1.0;
const EPSILON_END = 0.05;
const EPSILON_DECAY = 0.999;
const EVAL_EPISODES = 200;
const SEED = 0;

function epsilonGreedy(Q: number[][], state: number, epsilon: number, nActions: number, rng: () => number): number {
  if (rng() < epsilon) {
    return Math.floor(rng() * nActions);
  }
  let maxVal = Q[state][0];
  let bestAction = 0;
  for (let a = 1; a < nActions; a++) {
    if (Q[state][a] > maxVal) {
      maxVal = Q[state][a];
      bestAction = a;
    }
  }
  return bestAction;
}

export function trainFrozenLake(): void {
  const env = make('FrozenLake-v1');
  const recorder = new RecordEpisodeWrapper(env);

  const nStates = (env as any).observationSpace?.n ?? 16;
  const nActions = (env as any).actionSpace?.n ?? 4;
  const Q: number[][] = Array.from({ length: nStates }, () => new Array(nActions).fill(0));

  let epsilon = EPSILON_START;
  const rng = () => Math.random();

  console.log('='.repeat(55));
  console.log('  world  ·  FrozenLake-v1  ·  Q-Learning');
  console.log('='.repeat(55));

  for (let ep = 0; ep < EPISODES; ep++) {
    const resetResult = recorder.reset({ seed: Math.floor(rng() * 1_000_000) });
    let obs = resetResult.observation as number;

    for (let step = 0; step < MAX_STEPS; step++) {
      const action = epsilonGreedy(Q, obs, epsilon, nActions, rng);
      const result = recorder.step(action);
      const nextObs = result.observation as number;
      const reward = result.reward;

      const bestNext = Math.max(...Q[nextObs]);
      Q[obs][action] += ALPHA * (reward + GAMMA * bestNext - Q[obs][action]);

      obs = nextObs;
      if (result.done) break;
    }

    epsilon = Math.max(EPSILON_END, epsilon * EPSILON_DECAY);

    if ((ep + 1) % 500 === 0) {
      const stats = recorder.summary();
      const last500 = recorder.episode_stats.slice(-500);
      const wins = last500.filter(e => e.reward > 0).length;
      const winRate = wins / last500.length;
      console.log(`  Episode ${String(ep + 1).padStart(5)} | ε=${epsilon.toFixed(3)} | win_rate(500)=${(winRate * 100).toFixed(0)}%`);
    }
  }

  console.log('\n  Training complete!');
  console.log(`  Total episodes recorded: ${recorder.episode_stats.length}`);

  // Evaluation (greedy)
  console.log(`\n  Evaluating greedy policy over ${EVAL_EPISODES} episodes …`);
  let wins = 0;
  const evalEnv = make('FrozenLake-v1');
  for (let ep = 0; ep < EVAL_EPISODES; ep++) {
    const resetResult = evalEnv.reset({ seed: ep });
    let obs = resetResult.observation as number;
    for (let step = 0; step < MAX_STEPS; step++) {
      const action = Q[obs].indexOf(Math.max(...Q[obs]));
      const result = evalEnv.step(action);
      obs = result.observation as number;
      if (result.done) {
        if (result.reward > 0) wins++;
        break;
      }
    }
  }
  evalEnv.close();
  console.log(`  Win rate: ${wins}/${EVAL_EPISODES} = ${((wins / EVAL_EPISODES) * 100).toFixed(0)}%`);
  console.log('='.repeat(55));

  // Demo render
  console.log('\n  Rendering one greedy episode:\n');
  const demoEnv = make('FrozenLake-v0');
  const demoReset = demoEnv.reset({ seed: SEED });
  let obs = demoReset.observation as number;
  demoEnv.render();
  for (let step = 0; step < MAX_STEPS; step++) {
    const action = Q[obs].indexOf(Math.max(...Q[obs]));
    const result = demoEnv.step(action);
    console.log(`\n  → action=${action}  reward=${result.reward}  done=${result.done}`);
    demoEnv.render();
    obs = result.observation as number;
    if (result.done) break;
  }
  demoEnv.close();
}

trainFrozenLake();