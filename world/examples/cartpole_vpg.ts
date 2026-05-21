/**
 * examples/cartpole_vpg.ts
 * VPG (Vanilla Policy Gradient / REINFORCE) for CartPole-v1
 * using ai4 nn framework with numerical gradients.
 *
 * The policy network is a 3-layer MLP: 4 → 128 → 64 → 2 (softmax).
 * Training uses REINFORCE with discounted returns and Adam optimizer.
 *
 * Usage:
 *   npx tsx world/examples/cartpole_vpg.ts
 */
import { CartPoleEnv } from '../envs/cartpole';
import { Tensor, Module, Linear, ReLU, Sequential, Adam } from '../../nn/index';

class PolicyNet extends Module {
  net: Sequential;

  constructor(obsDim: number, actionN: number) {
    super();
    this.net = new Sequential([
      new Linear(obsDim, 128, true),
      new ReLU(),
      new Linear(128, 64, true),
      new ReLU(),
      new Linear(64, actionN, true),
    ]);
  }

  forward(x: Tensor): Tensor {
    return this.net.forward(x);
  }
}

function softmax1d(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map(v => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(v => v / (sum + 1e-10));
}

function sample(probs: number[]): number {
  const r = Math.random();
  let cum = 0;
  for (let i = 0; i < probs.length; i++) {
    cum += probs[i];
    if (r <= cum) return i;
  }
  return probs.length - 1;
}

function discountedReturns(rewards: number[], gamma: number): number[] {
  const returns = new Array(rewards.length);
  let G = 0;
  for (let t = rewards.length - 1; t >= 0; t--) {
    G = rewards[t] + gamma * G;
    returns[t] = G;
  }
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.reduce((s, r) => s + (r - mean) ** 2, 0) / returns.length) + 1e-8;
  return returns.map(r => (r - mean) / std);
}

function computeLoss(
  policyNet: PolicyNet,
  states: number[][],
  actions: number[],
  returns: number[],
): number {
  const T = states.length;
  const flatStates = states.flat();
  const stateTensor = new Tensor(flatStates, [T, 4], false);
  const logits = policyNet.forward(stateTensor);
  const probs = logits.softmax(1);
  let loss = 0;
  for (let t = 0; t < T; t++) {
    const p = probs.data[t * 2 + actions[t]];
    loss -= returns[t] * Math.log(Math.max(p, 1e-10));
  }
  return loss / T;
}

async function playEpisode(
  env: CartPoleEnv,
  policyNet: PolicyNet,
  seed?: number,
  mode?: 'train' | 'eval',
): Promise<{ reward: number; steps: number; states?: number[][]; actions?: number[]; rewards?: number[] }> {
  const { observation } = env.reset({ seed });
  const states: number[][] = [];
  const actions: number[] = [];
  const rewards: number[] = [];
  let obs = observation;
  let totalReward = 0;
  let steps = 0;
  const done = false;

  while (!done) {
    const logits = policyNet.forward(new Tensor(obs, [1, 4], false));
    const probs = softmax1d(logits.data);
    const action = mode === 'eval' ? probs.indexOf(Math.max(...probs)) : sample(probs);

    const result = env.step(action);
    obs = result.observation as number[];

    if (mode === 'train') {
      states.push([...obs]);
      actions.push(action);
      rewards.push(result.reward);
    }

    totalReward += result.reward;
    steps++;

    if (result.terminated || result.truncated) break;
  }

  return mode === 'train'
    ? { reward: totalReward, steps, states, actions, rewards }
    : { reward: totalReward, steps };
}

function learn(
  policyNet: PolicyNet,
  optimizer: Adam,
  states: number[][],
  actions: number[],
  rewards: number[],
  gamma: number,
): void {
  const returns = discountedReturns(rewards, gamma);
  const params = policyNet.parameters();
  const eps = 1e-5;

  for (const p of params) {
    const grad = new Array(p.data.length).fill(0);
    const original = [...p.data];

    for (let i = 0; i < p.data.length; i++) {
      p.data[i] = original[i] + eps;
      const lossPlus = computeLoss(policyNet, states, actions, returns);

      p.data[i] = original[i] - eps;
      const lossMinus = computeLoss(policyNet, states, actions, returns);

      grad[i] = (lossPlus - lossMinus) / (2 * eps);
      p.data[i] = original[i];
    }

    p.grad = grad;
  }

  optimizer.step();
  optimizer.zeroGrad();
}

async function main(): Promise<void> {
  const env = new CartPoleEnv();
  const policyNet = new PolicyNet(4, 2);
  const optimizer = new Adam(policyNet.parameters(), 0.005);
  const gamma = 0.99;
  const maxEpisodes = 1000;

  console.log('='.repeat(55));
  console.log('  CartPole-v1  ·  VPG (REINFORCE)  ·  ai4 nn framework');
  console.log('='.repeat(55));

  // Training
  console.log('\n=== Training ===');
  const episodeRewards: number[] = [];

  for (let ep = 0; ep < maxEpisodes; ep++) {
    const { reward, steps, states, actions, rewards } = await playEpisode(env, policyNet, ep * 100, 'train');

    if (states && actions && rewards) {
      learn(policyNet, optimizer, states, actions, rewards, gamma);
    }

    episodeRewards.push(reward);

    if (ep % 50 === 0 || ep === maxEpisodes - 1) {
      console.log(`  episode ${ep}: reward = ${reward.toFixed(2)}, steps = ${steps}`);
    }

    if (episodeRewards.length >= 20) {
      const avg = episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
      if (avg > 199) {
        console.log(`\n  Solved at episode ${ep}! Average reward (last 20): ${avg.toFixed(2)}`);
        break;
      }
    }
  }

  // Test
  console.log('\n=== Testing (20 episodes) ===');
  const testRewards: number[] = [];
  for (let i = 0; i < 20; i++) {
    const { reward } = await playEpisode(env, policyNet, i * 1000 + 999, 'eval');
    testRewards.push(reward);
    console.log(`  test ${i + 1}: reward = ${reward.toFixed(2)}`);
  }

  const mean = testRewards.reduce((a, b) => a + b, 0) / testRewards.length;
  const std = Math.sqrt(testRewards.reduce((s, r) => s + (r - mean) ** 2, 0) / testRewards.length);
  console.log(`\n  average test reward = ${mean.toFixed(2)} ± ${std.toFixed(2)}`);
  console.log('='.repeat(55));

  env.close();
}

main().catch(console.error);
