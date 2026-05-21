/**
 * examples/cartpole_closed_form.ts
 * CartPole-v1 with a closed-form (heuristic) PD-like controller.
 *
 * The controller computes force direction from theta and theta_dot:
 *   if theta > 0 (falling right) → push right
 *   if theta < 0 (falling left)  → push left
 *   theta_dot adds damping to reduce overshoot.
 *
 * Usage:
 *   npx tsx world/examples/cartpole_closed_form.ts
 */
import { CartPoleEnv } from '../envs/cartpole';

async function main(): Promise<void> {
  const env = new CartPoleEnv();
  const episodes = 10;
  const maxSteps = 500;
  const render = process.argv.includes('--render');

  console.log('='.repeat(50));
  console.log('  CartPole-v1  ·  Closed-Form (Heuristic) Controller');
  if (render) console.log('  Render mode: browser (http://localhost:8080)');
  console.log('='.repeat(50));

  if (render) {
    console.log('  Waiting for browser connection (3s)...');
    await new Promise(r => setTimeout(r, 3000));
  }

  let totalSteps = 0;

  for (let ep = 0; ep < episodes; ep++) {
    const { observation } = env.reset({ seed: ep * 100 });
    let obs = observation;
    let steps = 0;

    for (let s = 0; s < maxSteps; s++) {
      const [, , theta, theta_dot] = obs;
      const action = theta > 0
        ? (theta_dot > 0.01 ? 1 : 0)
        : (theta_dot < -0.01 ? 0 : 1);

      const result = env.step(action);
      obs = result.observation as number[];
      steps++;

      if (render) {
        env.render('human');
        await new Promise(r => setTimeout(r, 33));
      }
      if (result.terminated || result.truncated) break;
    }

    totalSteps += steps;
    console.log(`  Episode ${ep + 1}: ${steps} steps`);
  }

  console.log('='.repeat(50));
  console.log(`  Average: ${(totalSteps / episodes).toFixed(1)} steps`);
  console.log('='.repeat(50));

  if (render) {
    console.log('  Press Enter to stop server...');
    process.stdin.resume();
    await new Promise<void>(resolve => process.stdin.once('data', () => resolve()));
  }
  env.close();
}

main().catch(console.error);
