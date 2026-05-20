import { DecisionTreeClassifier, DecisionTreeRegressor } from './tree';

export class RandomForestClassifier {
  trees: DecisionTreeClassifier[] = [];
  n_estimators: number;
  max_depth: number;

  constructor(n_estimators = 10, max_depth = 10) {
    this.n_estimators = n_estimators;
    this.max_depth = max_depth;
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];
    const m = X.length;
    for (let i = 0; i < this.n_estimators; i++) {
      const idx = Array.from({ length: m }, () => Math.floor(Math.random() * m));
      const X_boot = idx.map(i => X[i]);
      const y_boot = idx.map(i => y[i]);
      const tree = new DecisionTreeClassifier(this.max_depth);
      tree.fit(X_boot, y_boot);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    const votes: Record<number, number>[] = X.map(() => ({}));
    for (const tree of this.trees) {
      const pred = tree.predict(X);
      for (let i = 0; i < pred.length; i++) {
        (votes[i] as Record<number, number>)[pred[i]] = ((votes[i] as Record<number, number>)[pred[i]] || 0) + 1;
      }
    }
    return votes.map(v => Number(Object.entries(v).sort((a, b) => b[1] - a[1])[0]?.[0] ?? 0));
  }
}

export class RandomForestRegressor {
  trees: DecisionTreeRegressor[] = [];
  n_estimators: number;
  max_depth: number;

  constructor(n_estimators = 10, max_depth = 10) {
    this.n_estimators = n_estimators;
    this.max_depth = max_depth;
  }

  fit(X: number[][], y: number[]): void {
    this.trees = [];
    const m = X.length;
    for (let i = 0; i < this.n_estimators; i++) {
      const idx = Array.from({ length: m }, () => Math.floor(Math.random() * m));
      const X_boot = idx.map(i => X[i]);
      const y_boot = idx.map(i => y[i]);
      const tree = new DecisionTreeRegressor(this.max_depth);
      tree.fit(X_boot, y_boot);
      this.trees.push(tree);
    }
  }

  predict(X: number[][]): number[] {
    const preds = this.trees.map(t => t.predict(X));
    return X.map((_, i) => preds.reduce((s, p) => s + p[i], 0) / preds.length);
  }
}

export class GradientBoostingClassifier {
  lr: number;
  n_estimators: number;
  trees: DecisionTreeRegressor[] = [];
  initial_pred: number = 0;

  constructor(lr = 0.1, n_estimators = 10) {
    this.lr = lr;
    this.n_estimators = n_estimators;
  }

  fit(X: number[][], y: number[]): void {
    const pred = new Array(y.length).fill(0.5);
    this.initial_pred = 0.5;
    this.trees = [];
    for (let t = 0; t < this.n_estimators; t++) {
      const residual = y.map((yi, i) => yi - pred[i]);
      const tree = new DecisionTreeRegressor(3);
      tree.fit(X, residual);
      const update = tree.predict(X);
      for (let i = 0; i < pred.length; i++) pred[i] += this.lr * update[i];
      this.trees.push(tree);
    }
  }

  sigmoid(z: number): number {
    return 1 / (1 + Math.exp(-Math.min(500, Math.max(-500, z))));
  }

  predict_proba(X: number[][]): number[] {
    let pred = new Array(X.length).fill(this.initial_pred);
    for (const tree of this.trees) {
      const update = tree.predict(X);
      for (let i = 0; i < pred.length; i++) pred[i] += this.lr * update[i];
    }
    return pred.map(p => this.sigmoid(p));
  }

  predict(X: number[][]): number[] {
    return this.predict_proba(X).map(p => (p > 0.5 ? 1 : 0));
  }
}