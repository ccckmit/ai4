import { Env, StepResult } from '../core';

export class TimeLimitWrapper<ObsType, ActType> extends Env<ObsType, ActType> {
  private _env: Env<ObsType, ActType>;
  private _max_steps: number;
  private _elapsed = 0;

  get observationSpace() { return this._env.observationSpace; }
  get actionSpace() { return this._env.actionSpace; }

  constructor(env: Env<ObsType, ActType>, max_steps: number) {
    super();
    this._env = env;
    this._max_steps = max_steps;
  }

  reset(options?: { seed?: number }) {
    this._elapsed = 0;
    return this._env.reset(options);
  }

  step(action: ActType): StepResult<ObsType> {
    const result = this._env.step(action);
    this._elapsed++;
    if (this._elapsed >= this._max_steps && !result.terminated) {
      return new StepResult(result.observation, result.reward, false, true, result.info);
    }
    return result;
  }

  render() { return this._env.render(); }
  close() { this._env.close(); }
}