import { Env, StepResult } from '../core';
import { Discrete } from '../spaces/discrete';
import { sendFrame } from '../render/server';

const GRAVITY = 9.8;
const MASS_CART = 1.0;
const MASS_POLE = 0.1;
const TOTAL_MASS = MASS_CART + MASS_POLE;
const LENGTH = 0.5;
const POLE_MASS_LENGTH = MASS_POLE * LENGTH;
const FORCE_MAG = 10.0;
const TAU = 0.02;

const THETA_THRESHOLD = 12 * Math.PI / 180;
const X_THRESHOLD = 2.4;

export class CartPoleEnv extends Env<number[], number> {
  readonly observationSpace: Discrete;
  readonly actionSpace: Discrete;

  private _x = 0;
  private _x_dot = 0;
  private _theta = 0;
  private _theta_dot = 0;
  _steps = 0;
  _max_steps = 500;

  constructor() {
    super();
    this.observationSpace = new Discrete(4);
    this.actionSpace = new Discrete(2);
  }

  reset(options?: { seed?: number }): { observation: number[]; info: Record<string, unknown> } {
    this.initRNG(options?.seed);
    this._x = 0;
    this._x_dot = 0;
    this._theta = 0;
    this._theta_dot = 0;
    this._steps = 0;
    return {
      observation: [this._x, this._x_dot, this._theta, this._theta_dot],
      info: {},
    };
  }

  step(action: number): StepResult<number[]> {
    this._steps++;
    const force = action === 1 ? FORCE_MAG : -FORCE_MAG;

    const cosTheta = Math.cos(this._theta);
    const sinTheta = Math.sin(this._theta);

    const temp = (force + POLE_MASS_LENGTH * this._theta_dot ** 2 * sinTheta) / TOTAL_MASS;
    const thetaAcc =
      (GRAVITY * sinTheta - cosTheta * temp) /
      (LENGTH * (4 / 3 - MASS_POLE * cosTheta ** 2 / TOTAL_MASS));
    const xAcc = temp - POLE_MASS_LENGTH * thetaAcc * cosTheta / TOTAL_MASS;

    this._x += TAU * this._x_dot;
    this._x_dot += TAU * xAcc;
    this._theta += TAU * this._theta_dot;
    this._theta_dot += TAU * thetaAcc;

    const terminated =
      this._x < -X_THRESHOLD ||
      this._x > X_THRESHOLD ||
      this._theta < -THETA_THRESHOLD ||
      this._theta > THETA_THRESHOLD;

    const truncated = !terminated && this._steps >= this._max_steps;
    const reward = terminated ? 0.0 : 1.0;

    return new StepResult(
      [this._x, this._x_dot, this._theta, this._theta_dot],
      reward,
      terminated,
      truncated,
      {
        x: this._x,
        theta_deg: this._theta * 180 / Math.PI,
        steps: this._steps,
      }
    );
  }

  render(mode: 'ansi' | 'human' = 'ansi'): void {
    if (mode === 'human') {
      sendFrame({
        x: this._x,
        theta: this._theta,
        steps: this._steps,
        reward: 0,
        done: this._steps >= this._max_steps,
      });
    } else {
      console.log('x=' + this._x.toFixed(3) + ' θ=' + (this._theta * 180 / Math.PI).toFixed(1) + '°');
    }
  }
}