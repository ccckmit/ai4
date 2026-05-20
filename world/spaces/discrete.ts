import { Space } from '../core';

export class Discrete extends Space {
  readonly n: number;
  private readonly _start: number;

  constructor(n: number, start = 0) {
    super();
    this.n = n;
    this._start = start;
  }

  sample(): number {
    return Math.floor(Math.random() * this.n) + this._start;
  }

  contains(x: number): boolean {
    return Number.isInteger(x) && x >= this._start && x < this._start + this.n;
  }
}