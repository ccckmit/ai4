/**
 * ai4 world - Reinforcement Learning environments
 * Core abstractions: Env, StepResult, Space, MathRandomGenerator
 */

export class StepResult<ObsType> {
  observation: ObsType;
  reward: number;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, unknown>;

  constructor(
    observation: ObsType,
    reward: number,
    terminated: boolean,
    truncated: boolean,
    info: Record<string, unknown> = {}
  ) {
    this.observation = observation;
    this.reward = reward;
    this.terminated = terminated;
    this.truncated = truncated;
    this.info = info;
  }

  get done(): boolean {
    return this.terminated || this.truncated;
  }
}

export abstract class Space {
  abstract get n(): number;
  sample(): number | number[] {
    return 0;
  }
}

export abstract class Env<ObsType = unknown, ActType = unknown> {
  metadata: Record<string, unknown> = { render_modes: ['ansi'] };
  rewardRange: [number, number] = [-Infinity, Infinity];
  protected _np_random?: MathRandomGenerator;
  protected _steps = 0;

  abstract get observationSpace(): Space;
  abstract get actionSpace(): Space;

  abstract reset(options?: { seed?: number }): { observation: ObsType; info: Record<string, unknown> };
  abstract step(action: ActType): StepResult<ObsType>;

  render(): unknown { return null; }
  close(): void {}

  seed(seed?: number): number[] {
    this._np_random = new MathRandomGenerator(seed);
    return [seed ?? 0];
  }

  protected initRNG(seed?: number): MathRandomGenerator {
    this._np_random = new MathRandomGenerator(seed);
    return this._np_random;
  }

  protected get _rng(): MathRandomGenerator {
    return this._np_random ?? new MathRandomGenerator();
  }
}

export class MathRandomGenerator {
  private seed: number;

  constructor(seed?: number) {
    this.seed = seed ?? Math.floor(Math.random() * 0x7fffffff);
  }

  random(): number {
    this.seed = (this.seed * 1103515245 + 12345) & 0x7fffffff;
    return this.seed / 0x7fffffff;
  }

  integers(min: number, max: number): number {
    return Math.floor(min + this.random() * (max - min));
  }

  choice<T>(arr: T[]): T {
    return arr[Math.floor(this.random() * arr.length)];
  }

  shuffle<T>(arr: T[]): T[] {
    const result = [...arr];
    for (let i = result.length - 1; i > 0; i--) {
      const j = Math.floor(this.random() * (i + 1));
      [result[i], result[j]] = [result[j], result[i]];
    }
    return result;
  }
}