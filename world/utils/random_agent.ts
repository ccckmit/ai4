import { Env, StepResult } from '../core';

export function run_random_agent(env: Env, episodes = 10): void {
  for (let ep = 0; ep < episodes; ep++) {
    const { info } = env.reset({ seed: ep });
    let total = 0;
    while (true) {
      const action = env.actionSpace.sample();
      const result = env.step(action) as StepResult<unknown>;
      total += result.reward;
      if (result.done) break;
    }
    console.log(`Episode ${ep}: reward=${total}`);
  }
}

export function run_pd_agent(
  env: Env,
  controller: { act: (obs: number[]) => number },
  episodes = 10
): void {
  for (let ep = 0; ep < episodes; ep++) {
    const { observation: obs } = env.reset({ seed: ep });
    let total = 0;
    while (true) {
      const action = controller.act(obs as number[]);
      const result = env.step(action) as StepResult<number[]>;
      total += result.reward;
      if (result.done) break;
    }
    console.log(`Episode ${ep}: reward=${total}`);
  }
}