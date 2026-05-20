export class StandardScaler {
  mean: number[] = [];
  std: number[] = [];

  fit(X: number[][]): void {
    const n = X.length;
    this.mean = X[0].map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
    this.std = X[0].map((_, j) => {
      const v = Math.sqrt(X.reduce((s, r) => s + (r[j] - this.mean[j]) ** 2, 0) / n);
      return v === 0 ? 1 : v;
    });
  }

  transform(X: number[][]): number[][] {
    return X.map(r => r.map((v, j) => (v - this.mean[j]) / this.std[j]));
  }

  fit_transform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }
}

export class MinMaxScaler {
  min: number[] = [];
  max: number[] = [];

  fit(X: number[][]): void {
    const n = X.length;
    this.min = X[0].map((_, j) => Math.min(...X.map(r => r[j])));
    this.max = X[0].map((_, j) => Math.max(...X.map(r => r[j])));
  }

  transform(X: number[][]): number[][] {
    return X.map(r => r.map((v, j) => {
      const range = this.max[j] - this.min[j];
      return range === 0 ? 0 : (v - this.min[j]) / range;
    }));
  }

  fit_transform(X: number[][]): number[][] {
    this.fit(X);
    return this.transform(X);
  }
}

export function train_test_split(
  X: number[][],
  y: number[],
  test_size = 0.2,
  seed?: number
): { X_train: number[][]; X_test: number[][]; y_train: number[]; y_test: number[] } {
  if (seed !== undefined) {
    let s = seed;
    const rand = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
    const idx = Array.from({ length: X.length }, (_, i) => i).sort(() => rand() - 0.5);
    const t = Math.floor(X.length * test_size);
    const testIdx = idx.slice(0, t);
    const trainIdx = idx.slice(t);
    return {
      X_train: trainIdx.map(i => X[i]),
      X_test: testIdx.map(i => X[i]),
      y_train: trainIdx.map(i => y[i]),
      y_test: testIdx.map(i => y[i]),
    };
  }
  const idx = Array.from({ length: X.length }, (_, i) => i).sort(() => Math.random() - 0.5);
  const t = Math.floor(X.length * test_size);
  const testIdx = idx.slice(0, t);
  const trainIdx = idx.slice(t);
  return {
    X_train: trainIdx.map(i => X[i]),
    X_test: testIdx.map(i => X[i]),
    y_train: trainIdx.map(i => y[i]),
    y_test: testIdx.map(i => y[i]),
  };
}