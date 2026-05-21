import { CartPoleEnv } from '../envs/cartpole';

const env = new CartPoleEnv();
const episodes = 3;

for (let ep = 0; ep < episodes; ep++) {
  const { observation, info } = env.reset({ seed: ep * 100 });
  let [x, xd, theta, thd] = observation;
  let done = false;
  let total = 0;

  while (!done) {
    // simple PD controller
    const force = theta > 0.1 ? 0 : theta < -0.1 ? 1 : Math.random() < 0.5 ? 0 : 1;
    const result = env.step(force);
    [x, xd, theta, thd] = result.observation;
    total += result.reward;
    done = result.terminated || result.truncated;
    env.render('human');
    // slow down so browser can see each frame
    await new Promise(r => setTimeout(r, 20));
  }
  console.log(`Episode ${ep + 1}: reward = ${total}`);
}

console.log('Done — close the browser tab and press Ctrl+C to stop');
