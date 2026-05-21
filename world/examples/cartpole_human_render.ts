/**
 * examples/cartpole_human_render.ts
 * CartPole-v1 with PD controller and live browser rendering via WebSocket.
 *
 * Usage:
 *   npx tsx world/examples/cartpole_human_render.ts
 */
import { CartPoleEnv } from '../envs/cartpole';

class PDController {
  constructor(private kp = 25.0, private kd = 5.0) {}
  act(obs: number[]): number {
    const [, , theta, theta_dot] = obs;
    return this.kp * theta + this.kd * theta_dot > 0 ? 1 : 0;
  }
}

async function main(): Promise<void> {
  const env = new CartPoleEnv();
  const controller = new PDController();
  const episodes = 5;

  console.log('Opening browser at http://localhost:8080 ...');
  console.log(`Running ${episodes} episodes...`);

  for (let ep = 0; ep < episodes; ep++) {
    const { observation } = env.reset({ seed: ep * 100 });
    let obs = observation;
    let total = 0;
    let done = false;
    let steps = 0;

    while (!done) {
      const action = controller.act(obs);
      const result = env.step(action);
      obs = result.observation;
      total += result.reward;
      done = result.terminated || result.truncated;
      steps++;
      env.render('human');
      await new Promise(r => setTimeout(r, 16)); // ~60fps
    }
    console.log(`Episode ${ep + 1}: ${steps} steps, reward = ${total}`);
  }

  console.log('Done! Press Ctrl+C to stop the server.');
}

main().catch(console.error);
