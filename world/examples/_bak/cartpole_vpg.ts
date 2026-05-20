/**
 * VPG (Vanilla Policy Gradient / REINFORCE) implementation for CartPole
 * Uses world/CartPole-v1 environment and nn/ tensor autodiff
 * Reference: https://zhiqingxiao.github.io/rl-book/en2024/code/CartPole-v0_VPG_torch.html
 */
import { make } from '../index';
import { Tensor, Module, Linear } from '../../nn/index';

class VPGAgent {
  action_n: number;
  gamma = 0.99;
  policy_net: Module;
  optimizer: { step: () => void; zeroGrad: () => void; lr: number };

  constructor(env: { observationSpace: { shape?: number[] }; action_space: { n: number } }) {
    this.action_n = env.action_space.n;
    const obs_dim = env.observationSpace.shape?.[0] ?? 4;

    this.policy_net = new PolicyNet(obs_dim, this.action_n);
    this.optimizer = new (await import('../../nn/index')).Adam(this.policy_net.parameters(), 0.005);
  }

  mode: string | null = null;
  trajectory: number[] = [];

  reset(mode?: string): void {
    this.mode = mode ?? null;
    if (this.mode === 'train') {
      this.trajectory = [];
    }
  }

  step(observation: number[], reward: number, terminated: boolean): number {
    const stateTensor = Tensor.from([observation]);
    const probTensor = (this.policy_net as any).forward(stateTensor) as Tensor;
    const probs = probTensor.data[0] ?? [];
    const action = this.sampleAction(probs);

    if (this.mode === 'train') {
      this.trajectory.push(...observation, reward, terminated ? 1 : 0, action);
    }
    return action;
  }

  private sampleAction(probs: number[]): number {
    const r = Math.random();
    let cumsum = 0;
    for (let i = 0; i < probs.length; i++) {
      cumsum += probs[i];
      if (r <= cumsum) return i;
    }
    return probs.length - 1;
  }

  close(): void {
    if (this.mode === 'train') {
      this.learn();
    }
  }

  learn(): void {
    const traj = this.trajectory;
    const n = Math.floor(traj.length / 4);
    const states: number[][] = [];
    const rewards: number[] = [];
    const actions: number[] = [];

    for (let i = 0; i < n; i++) {
      states.push(traj.slice(i * 4, i * 4 + 4));
      rewards.push(traj[i * 4 + 1]);
      actions.push(traj[i * 4 + 3]);
    }

    const stateTensor = Tensor.from(states, true);
    const rewardTensor = Tensor.from([rewards]);
    const actionTensor = Tensor.from([actions]);

    const discount = this.gamma;
    let cumulative: number[] = [];
    let G = 0;
    for (let t = n - 1; t >= 0; t--) {
      G = rewards[t] + discount * G;
      cumulative = [G, ...cumulative];
    }
    const discountedReturn = Tensor.from([cumulative]);

    const allPi = (this.policy_net as any).forward(stateTensor) as Tensor;
    const pi = actions.map((a, i) => allPi.data[i]?.[a] ?? 0);
    const logPi = pi.map(p => Math.log(Math.max(p, 1e-8)));
    let loss = 0;
    for (let t = 0; t < n; t++) {
      loss -= discountedReturn.data[0]?.[t] ?? 0 * logPi[t];
    }
    loss /= n;

    const lossTensor = Tensor.from([[loss]], false, true);
    const out = lossTensor;
    out.requires_grad = true;

    (this.policy_net as any)._loss = out;
    this.optimizer.zeroGrad();

    const params = this.policy_net.parameters();
    for (const p of params) {
      if (p.grad) {
        for (let i = 0; i < p.grad.length; i++) {
          for (let j = 0; j < p.grad[i].length; j++) {
            p.grad[i][j] = 0;
          }
        }
      }
    }

    const lr = this.optimizer.lr;
    for (const p of params) {
      if (p.data && p.grad) {
        for (let i = 0; i < p.data.length; i++) {
          for (let j = 0; j < p.data[i].length; j++) {
            const grad = (p.grad[i]?.[j] ?? 0) * lr;
            p.data[i][j] -= grad;
          }
        }
      }
    }
  }
}

class PolicyNet extends Module {
  linear: Linear;

  constructor(inputSize: number, outputSize: number) {
    super();
    this.linear = new Linear(inputSize, outputSize, false);
  }

  forward(x: Tensor): Tensor {
    const out = this.linear.forward(x);
    const probs = out.data.map((row: number[]) => {
      const maxLogit = Math.max(...row);
      const expRow = row.map(v => Math.exp(v - maxLogit));
      const sumExp = expRow.reduce((a, b) => a + b, 0);
      return expRow.map(v => v / sumExp);
    });
    return new Tensor(probs, [x], false);
  }
}

async function playEpisode(
  env: ReturnType<typeof make>,
  agent: VPGAgent,
  seed?: number,
  mode?: string
): Promise<{ reward: number; steps: number }> {
  const resetResult = env.reset({ seed });
  let observation = resetResult.observation as number[];
  let reward = 0;
  let terminated = false;
  agent.reset(mode);

  let episodeReward = 0;
  let elapsedSteps = 0;

  while (true) {
    const action = agent.step(observation, reward, terminated);
    if (terminated) break;

    const stepResult = env.step(action);
    observation = stepResult.observation as number[];
    reward = stepResult.reward;
    terminated = stepResult.terminated || stepResult.truncated;
    episodeReward += reward;
    elapsedSteps += 1;
  }
  agent.close();

  return { reward: episodeReward, steps: elapsedSteps };
}

async function main() {
  console.log('Creating CartPole environment...');
  const env = make('CartPole-v1');
  console.log(`Action space: ${env.actionSpace.n}`);

  const agent = new VPGAgent(env as any);

  console.log('\n=== Training ===');
  const episodeRewards: number[] = [];
  let episode = 0;
  const maxEpisodes = 1000;

  while (episode < maxEpisodes) {
    const { reward, steps } = await playEpisode(env, agent, episode, 'train');
    episodeRewards.push(reward);
    console.log(`episode ${episode}: reward = ${reward.toFixed(2)}, steps = ${steps}`);

    if (episode >= 19) {
      const avg = episodeRewards.slice(-20).reduce((a, b) => a + b, 0) / 20;
      if (avg > 199) {
        console.log(`\nSolved! Average reward over last 20 episodes: ${avg.toFixed(2)}`);
        break;
      }
    }
    episode++;
  }

  console.log('\n=== Testing (20 episodes) ===');
  const testRewards: number[] = [];
  for (let i = 0; i < 20; i++) {
    const { reward } = await playEpisode(env, agent);
    testRewards.push(reward);
    console.log(`test episode ${i}: reward = ${reward.toFixed(2)}`);
  }

  const meanReward = testRewards.reduce((a, b) => a + b, 0) / testRewards.length;
  const stdReward = Math.sqrt(
    testRewards.reduce((s, r) => s + (r - meanReward) ** 2, 0) / testRewards.length
  );
  console.log(`\naverage test reward = ${meanReward.toFixed(2)} ± ${stdReward.toFixed(2)}`);

  env.close();
}

main().catch(console.error);