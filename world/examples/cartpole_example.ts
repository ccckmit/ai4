/**
 * CartPole-v1 Random Agent & Heuristic Controller
 *
 * Demonstrates the CartPole environment with:
 * 1. Random agent baseline
 * 2. Heuristic (closed-form) balancing controller
 *
 * Run:
 *   npx tsx world/examples/cartpole_example.ts
 */
import { make } from '../index';

function runRandomAgent(episodes = 5): void {
  console.log('\n  Random agent:');
  for (let ep = 0; ep < episodes; ep++) {
    const env = make('CartPole-v1') as any;
    const resetResult = env.reset({ seed: ep });
    let obs = resetResult.observation as number[];
    let totalReward = 0;
    let steps = 0;
    let done = false;
    while (!done && steps < 500) {
      const action = env.actionSpace.sample() as number;
      const result = env.step(action);
      totalReward += result.reward;
      steps++;
      done = result.done;
      obs = result.observation as number[];
    }
    env.close();
    console.log(`    Episode ${ep + 1}: reward=${totalReward}, steps=${steps}`);
  }
}

function runHeuristicController(episodes = 3): void {
  console.log('\n  Heuristic controller:');
  for (let ep = 0; ep < episodes; ep++) {
    const env = make('CartPole-v1') as any;
    const resetResult = env.reset({ seed: ep });
    let obs = resetResult.observation as number[];
    let totalReward = 0;
    let steps = 0;
    let done = false;
    while (!done && steps < 500) {
      const angle = obs[2];
      const angularVelocity = obs[3];
      let action: number;
      if (angle > 0) {
        action = angularVelocity < -0.01 ? 0 : 1;
      } else {
        action = angularVelocity > 0.01 ? 1 : 0;
      }
      const result = env.step(action);
      totalReward += result.reward;
      steps++;
      done = result.done;
      obs = result.observation as number[];
    }
    env.close();
    console.log(`    Episode ${ep + 1}: reward=${totalReward}, steps=${steps}`);
  }
}

function main(): void {
  console.log('='.repeat(50));
  console.log('  CartPole-v1 Example');
  console.log('='.repeat(50));
  console.log('\n  State space: position, velocity, angle, angular_velocity');
  console.log('  Action space: 0=left, 1=right');
  console.log('  Goal: keep pole upright for 500 steps');

  runRandomAgent();
  runHeuristicController();

  console.log('\n' + '='.repeat(50));
  console.log('  Done!');
  console.log('='.repeat(50));
}

main();
