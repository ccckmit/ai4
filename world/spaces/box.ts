import { Space } from '../core';

export class Box extends Space {
  readonly shape: number[];
  readonly low: number[];
  readonly high: number[];
  readonly n: number;

  constructor(low: number, high: number, shape: number[]) {
    super();
    this.shape = shape;
    this.low = Array(shape.length).fill(low);
    this.high = Array(shape.length).fill(high);
    this.n = shape.reduce((a, b) => a * b, 1);
  }

  sample(): number[] {
    return Array.from({ length: this.shape[0] }, (_, i) =>
      this.low[i] + Math.random() * (this.high[i] - this.low[i])
    );
  }

  contains(x: number[]): boolean {
    if (x.length !== this.shape[0]) return false;
    for (let i = 0; i < x.length; i++) {
      if (x[i] < this.low[i] || x[i] > this.high[i]) return false;
    }
    return true;
  }
}