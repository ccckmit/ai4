type TreeNode = {
  feature: number;
  threshold: number;
  left: TreeNode;
  right: TreeNode;
} | {
  labels: number[];
};

function isLeaf(node: TreeNode): node is { labels: number[] } {
  return 'labels' in node;
}

export class DecisionTreeClassifier {
  max_depth: number;
  min_samples_split: number;
  n_features: number = 0;
  tree: TreeNode | null = null;

  constructor(max_depth = 10, min_samples_split = 2) {
    this.max_depth = max_depth;
    this.min_samples_split = min_samples_split;
  }

  gini(y: number[]): number {
    if (y.length === 0) return 0;
    const counts: Record<number, number> = {};
    for (const v of y) counts[v] = (counts[v] || 0) + 1;
    return 1 - Object.values(counts).reduce((s, c) => s + (c / y.length) ** 2, 0);
  }

  bestSplit(X: number[][], y: number[]): { feature: number; threshold: number; left_idx: number[]; right_idx: number[]; gain: number } | null {
    let best: { feature: number; threshold: number; left_idx: number[]; right_idx: number[]; gain: number } | null = null;
    const n = this.n_features || X[0].length;
    for (let f = 0; f < n; f++) {
      const vals = [...new Set(X.map(r => r[f]))].sort((a, b) => a - b);
      for (let i = 0; i < vals.length - 1; i++) {
        const t = (vals[i] + vals[i + 1]) / 2;
        const left_idx: number[] = [];
        const right_idx: number[] = [];
        for (let j = 0; j < X.length; j++) {
          if (X[j][f] <= t) left_idx.push(j); else right_idx.push(j);
        }
        if (left_idx.length === 0 || right_idx.length === 0) continue;
        const leftY = left_idx.map(idx => y[idx]);
        const rightY = right_idx.map(idx => y[idx]);
        const gain = this.gini(y) - (this.gini(leftY) * leftY.length / y.length) - (this.gini(rightY) * rightY.length / y.length);
        if (!best || gain > best.gain) {
          best = { feature: f, threshold: t, left_idx, right_idx, gain };
        }
      }
    }
    return best;
  }

  buildTree(X: number[][], y: number[], depth = 0): TreeNode | null {
    if (depth >= this.max_depth || y.length < this.min_samples_split) return { labels: y };
    const split = this.bestSplit(X, y);
    if (!split || split.gain <= 0) return { labels: y };

    const leftX = split.left_idx.map(i => X[i]);
    const rightX = split.right_idx.map(i => X[i]);
    const leftY = split.left_idx.map(i => y[i]);
    const rightY = split.right_idx.map(i => y[i]);

    const left = this.buildTree(leftX, leftY, depth + 1);
    const right = this.buildTree(rightX, rightY, depth + 1);

    if (!left || !right) return { labels: y };

    return { feature: split.feature, threshold: split.threshold, left, right };
  }

  fit(X: number[][], y: number[]): void {
    this.n_features = X[0].length;
    this.tree = this.buildTree(X, y);
  }

  predictOne(x: number[], node: TreeNode): number {
    if (isLeaf(node)) {
      const counts: Record<number, number> = {};
      for (const v of node.labels) counts[v] = (counts[v] || 0) + 1;
      return Number(Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0);
    }
    if (x[node.feature] <= node.threshold) return this.predictOne(x, node.left);
    return this.predictOne(x, node.right);
  }

  predict(X: number[][]): number[] {
    if (!this.tree) return [];
    return X.map(r => this.predictOne(r, this.tree!));
  }
}

export class DecisionTreeRegressor extends DecisionTreeClassifier {
  mse(y: number[]): number {
    if (y.length === 0) return 0;
    const m = y.reduce((s, v) => s + v, 0) / y.length;
    return y.reduce((s, v) => s + (v - m) ** 2, 0) / y.length;
  }

  predictOne(x: number[], node: TreeNode): number {
    if (isLeaf(node)) {
      return node.labels.reduce((s, v) => s + v, 0) / node.labels.length;
    }
    if (x[node.feature] <= node.threshold) return this.predictOne(x, node.left);
    return this.predictOne(x, node.right);
  }
}