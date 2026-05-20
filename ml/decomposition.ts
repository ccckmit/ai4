export class PCA {
  n_components: number;
  components: number[][] = [];
  mean: number[] = [];

  constructor(n_components = 2) {
    this.n_components = n_components;
  }

  fit(X: number[][]): void {
    const n = X.length;
    const m = X[0].length;
    this.mean = X[0].map((_, j) => X.reduce((s, r) => s + r[j], 0) / n);
    const Xc = X.map(r => r.map((v, j) => v - this.mean[j]));

    const cov: number[][] = [];
    for (let i = 0; i < m; i++) {
      cov[i] = [];
      for (let j = 0; j < m; j++) {
        let s = 0;
        for (let k = 0; k < n; k++) {
          s += Xc[k][i] * Xc[k][j];
        }
        cov[i][j] = s / (n - 1);
      }
    }

    const eigvecs: number[][] = [];
    for (let v = 0; v < this.n_components; v++) {
      const vec = new Array(m).fill(1 / Math.sqrt(m));
      for (let iter = 0; iter < 100; iter++) {
        const newVec = new Array(m).fill(0);
        for (let i = 0; i < m; i++) {
          for (let j = 0; j < m; j++) {
            newVec[i] += cov[i][j] * vec[j];
          }
        }
        const norm = Math.sqrt(newVec.reduce((s, v) => s + v * v, 0));
        if (norm < 1e-10) break;
        for (let i = 0; i < m; i++) vec[i] = newVec[i] / norm;
      }
      eigvecs.push(vec);

      for (let i = 0; i < m; i++) {
        for (let j = 0; j < m; j++) {
          cov[i][j] -= eigvecs[v][i] * eigvecs[v][j] * (cov[i]?.[j] ?? 0);
        }
      }
    }

    this.components = eigvecs;
  }

  transform(X: number[][]): number[][] {
    return X.map(r => {
      const centered = r.map((v, j) => v - this.mean[j]);
      return this.components.map(c => c.reduce((s, v, i) => s + v * centered[i], 0));
    });
  }
}