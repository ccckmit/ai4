export class KMeans {
  k: number;
  centers: number[][] = [];
  max_iter: number;

  constructor(k = 3, max_iter = 100) {
    this.k = k;
    this.max_iter = max_iter;
  }

  fit(X: number[][]): void {
    const n = X.length;
    const idx = Array.from({ length: n }, (_, i) => i).sort(() => Math.random() - 0.5).slice(0, this.k);
    this.centers = idx.map(i => [...X[i]]);

    for (let iter = 0; iter < this.max_iter; iter++) {
      const labels = X.map(x => {
        let best = 0, bestDist = Infinity;
        for (let c = 0; c < this.centers.length; c++) {
          const d = x.reduce((s, v, i) => s + (v - this.centers[c][i]) ** 2, 0);
          if (d < bestDist) { bestDist = d; best = c; }
        }
        return best;
      });

      const newCenters: number[][] = [];
      for (let c = 0; c < this.k; c++) {
        const pts = X.filter((_, i) => labels[i] === c);
        if (pts.length === 0) { newCenters.push(this.centers[c]); continue; }
        newCenters.push(pts[0].map((_, j) => pts.reduce((s, p) => s + p[j], 0) / pts.length));
      }
      if (this.centers.every((c, i) => c.every((v, j) => Math.abs(v - newCenters[i][j]) < 1e-6))) break;
      this.centers = newCenters;
    }
  }

  predict(X: number[][]): number[] {
    return X.map(x => {
      let best = 0, bestDist = Infinity;
      for (let c = 0; c < this.centers.length; c++) {
        const d = x.reduce((s, v, i) => s + (v - this.centers[c][i]) ** 2, 0);
        if (d < bestDist) { bestDist = d; best = c; }
      }
      return best;
    });
  }
}

export class DBSCAN {
  eps: number;
  min_samples: number;

  constructor(eps = 0.5, min_samples = 3) {
    this.eps = eps;
    this.min_samples = min_samples;
  }

  fit(X: number[][]): { labels: number[]; n_clusters: number } {
    const n = X.length;
    const labels = new Array(n).fill(-1);
    let cluster = 0;

    const neighbors = (i: number) => X.map((x, j) => (i !== j && x.reduce((s, v, k) => s + (v - X[i][k]) ** 2, 0) < this.eps ** 2 ? j : -1)).filter(v => v >= 0);

    const expand = (i: number, pts: number[]) => {
      labels[i] = cluster;
      const queue = [...pts];
      while (queue.length > 0) {
        const p = queue.shift()!;
        if (labels[p] === -1) labels[p] = cluster;
        const nb = neighbors(p);
        if (nb.length >= this.min_samples) {
          for (const nbItem of nb) {
            if (labels[nbItem] === -1) queue.push(nbItem);
            if (labels[nbItem] === -1 || labels[nbItem] === undefined) labels[nbItem] = cluster;
          }
        }
      }
    };

    for (let i = 0; i < n; i++) {
      if (labels[i] !== -1) continue;
      const nb = neighbors(i);
      if (nb.length >= this.min_samples) expand(i, nb);
      else labels[i] = -1;
    }

    return { labels, n_clusters: cluster + 1 };
  }
}