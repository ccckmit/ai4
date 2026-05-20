export class LinearRegression {
  weights: number[] = [];
  bias = 0;
  lr: number;
  n_iterations: number;

  constructor(lr = 0.01, n_iterations = 1000) {
    this.lr = lr;
    this.n_iterations = n_iterations;
  }

  fit(X: number[][], y: number[]): void {
    const m = X.length, n = X[0].length;
    this.weights = new Array(n).fill(0);
    this.bias = 0;
    for (let iter = 0; iter < this.n_iterations; iter++) {
      const pred = X.map(row => this.bias + this.weights.reduce((s, w, i) => s + w * row[i], 0));
      const error = pred.map((p, i) => p - y[i]);
      const dw = X[0].map((_, j) => this.lr * error.reduce((s, e, i) => s + e * X[i][j], 0) / m);
      this.bias -= this.lr * error.reduce((s, e) => s + e, 0) / m;
      this.weights = this.weights.map((w, j) => w - dw[j]);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => this.bias + this.weights.reduce((s, w, i) => s + w * row[i], 0));
  }
}

export class LogisticRegression {
  weights: number[] = [];
  bias = 0;
  lr: number;
  n_iterations: number;

  constructor(lr = 0.1, n_iterations = 1000) {
    this.lr = lr;
    this.n_iterations = n_iterations;
  }

  sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-Math.min(500, Math.max(-500, z))));
  }

  fit(X: number[][], y: number[]): void {
    const m = X.length, n = X[0].length;
    this.weights = new Array(n).fill(0);
    this.bias = 0;
    for (let iter = 0; iter < this.n_iterations; iter++) {
      const pred = X.map(row => this.sigmoid(this.bias + this.weights.reduce((s, w, i) => s + w * row[i], 0)));
      const error = pred.map((p, i) => p - y[i]);
      const dw = X[0].map((_, j) => this.lr * error.reduce((s, e, i) => s + e * X[i][j], 0) / m);
      this.bias -= this.lr * error.reduce((s, e) => s + e, 0) / m;
      this.weights = this.weights.map((w, j) => w - dw[j]);
    }
  }

  predict(X: number[][]): number[] {
    return X.map(row => (this.sigmoid(this.bias + this.weights.reduce((s, w, i) => s + w * row[i], 0)) > 0.5 ? 1 : 0));
  }

  predict_proba(X: number[][]): number[] {
    return X.map(row => this.sigmoid(this.bias + this.weights.reduce((s, w, i) => s + w * row[i], 0)));
  }
}