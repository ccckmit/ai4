/**
 * examples/cartpole_example.ts
 * Demonstrates CartPole-v1 with a simple PD controller + random agent comparison.
 */
import { make } from '../index';
import { RecordEpisodeWrapper } from '../wrappers/record_episode';

class PDController {
  constructor(
    private kp = 25.0,
    private kd = 5.0
  ) {}

  act(obs: number[]): number {
    const [, , theta, theta_dot] = obs;
    const signal = this.kp * theta + this.kd * theta_dot;
    return signal > 0 ? 1 : 0;
  }
}

function runPDAgent(episodes = 10, renderLast = true): void {
  const env = make('CartPole-v1');
  const recorder = new RecordEpisodeWrapper(env);

  console.log('='.repeat(55));
  console.log('  world  ·  CartPole-v1  ·  PD Controller');
  console.log('='.repeat(55));

  const controller = new PDController();

  for (let ep = 0; ep < episodes; ep++) {
    const resetResult = recorder.reset({ seed: ep });
    let obs = resetResult.observation as number[];
    while (true) {
      const action = controller.act(obs);
      const result = recorder.step(action);
      obs = result.observation as number[];
      if (result.done) break;
    }
  }

  const stats = recorder.summary();
  console.log(`  Episodes      : ${stats.episodes}`);
  console.log(`  Mean reward   : ${stats.mean_reward.toFixed(1)}`);
  console.log(`  Max  reward   : ${stats.max_reward.toFixed(1)}`);
  console.log(`  Mean length   : ${stats.mean_length.toFixed(1)}`);
  console.log('='.repeat(55));

  if (renderLast) {
    console.log('\n  Rendering final episode with PD controller:\n');
    const resetResult = env.reset({ seed: 999 });
    obs = resetResult.observation as number[];
    env.render();
    for (let i = 0; i < 500; i++) {
      const action = controller.act(obs);
      const result = env.step(action);
      obs = result.observation as number[];
      const actionStr = action === 1 ? '→' : '←';
      const x = (result.info as any)?.x ?? 0;
      const thetaDeg = (result.info as any)?.theta_deg ?? 0;
      console.log(`  action=${actionStr}  x=${x.toFixed(3)}  θ=${thetaDeg.toFixed(1)}°  reward=${result.reward.toFixed(0)}`);
      if (result.done) {
        const status = result.terminated ? 'TERMINATED' : 'TRUNCATED (max steps)';
        const steps = (result.info as any)?.steps ?? i + 1;
        console.log(`\n  Episode ended: ${status} after ${steps} steps`);
        break;
      }
    }
    env.render();
    env.close();
  }
}

function compareRandomVsPD(episodes = 20): void {
  console.log('\n' + '='.repeat(55));
  console.log('  Comparison: Random agent  vs  PD controller');
  console.log('='.repeat(55));

  // Random
  console.log('\n  [Random Agent]');
  const randomRewards: number[] = [];
  const env1 = make('CartPole-v1');
  for (let ep = 0; ep < episodes; ep++) {
    const resetResult = env1.reset({ seed: ep });
    let obs = resetResult.observation as number[];
    let total = 0;
    while (true) {
      const action = env1.actionSpace.sample();
      const result = env1.step(action);
      total += result.reward;
      obs = result.observation as number[];
      if (result.done) break;
    }
    randomRewards.push(total);
  }
  env1.close();
  const randomMean = randomRewards.reduce((a, b) => a + b, 0) / randomRewards.length;
  const randomStd = Math.sqrt(randomRewards.reduce((s, r) => s + (r - randomMean) ** 2, 0) / randomRewards.length);
  console.log(`  Mean reward: ${randomMean.toFixed(1)}  ±  ${randomStd.toFixed(1)}`);

  // PD
  console.log('\n  [PD Controller]');
  const pdRewards: number[] = [];
  const controller = new PDController();
  const env2 = make('CartPole-v1');
  for (let ep = 0; ep < episodes; ep++) {
    const resetResult = env2.reset({ seed: ep });
    let obs = resetResult.observation as number[];
    let total = 0;
    while (true) {
      const result = env2.step(controller.act(obs));
      total += result.reward;
      obs = result.observation as number[];
      if (result.done) break;
    }
    pdRewards.push(total);
  }
  env2.close();
  const pdMean = pdRewards.reduce((a, b) => a + b, 0) / pdRewards.length;
  const pdStd = Math.sqrt(pdRewards.reduce((s, r) => s + (r - pdMean) ** 2, 0) / pdRewards.length);
  console.log(`  Mean reward: ${pdMean.toFixed(1)}  ±  ${pdStd.toFixed(1)}`);
  console.log('='.repeat(55));
}

runPDAgent(10, true);
compareRandomVsPD(20);