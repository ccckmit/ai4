import { LinearRegression, LogisticRegression } from '../linear_models';
import { DecisionTree } from '../tree';
import { RandomForest } from '../ensemble';
import { KMeans } from '../clustering';
import { PCA } from '../decomposition';
import { StandardScaler, train_test_split } from '../preprocessing';
import { accuracy_score, mean_squared_error, r2_score } from '../metrics';

describe('LinearRegression', () => {
  test('fit and predict', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [2, 4, 6, 8, 10];
    const model = new LinearRegression(0.01, 1000);
    model.fit(X, y);
    const pred = model.predict(X);
    expect(pred.length).toBe(y.length);
    expect(pred.every((v, i) => Math.abs(v - y[i]) < 0.5)).toBe(true);
  });
});

describe('LogisticRegression', () => {
  test('binary classification', () => {
    const X = [[1], [2], [3], [4], [5], [6]];
    const y = [0, 0, 0, 1, 1, 1];
    const model = new LogisticRegression(0.1, 1000);
    model.fit(X, y);
    const pred = model.predict(X);
    expect(accuracy_score(y, pred)).toBeGreaterThan(0.5);
  });

  test('predict_proba sums to 1', () => {
    const X = [[0], [1], [2], [3]];
    const y = [0, 0, 1, 1];
    const model = new LogisticRegression(0.1, 500);
    model.fit(X, y);
    const proba = model.predict_proba(X);
    expect(proba.every(p => Math.abs(p[0] + p[1] - 1) < 0.01)).toBe(true);
  });
});

describe('DecisionTree', () => {
  test('classification', () => {
    const X = [[1], [2], [3], [4], [5], [6]];
    const y = [0, 0, 0, 1, 1, 1];
    const tree = new DecisionTree(3);
    tree.fit(X, y);
    const pred = tree.predict(X);
    expect(accuracy_score(y, pred)).toBeGreaterThan(0.5);
  });

  test('regression', () => {
    const X = [[1], [2], [3], [4], [5]];
    const y = [1, 1, 3, 3, 5];
    const tree = new DecisionTree(3);
    tree.fit(X, y);
    const pred = tree.predict(X);
    expect(mean_squared_error(y, pred)).toBeLessThan(1);
  });
});

describe('RandomForest', () => {
  test('classification', () => {
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < 100; i++) {
      const x0 = Math.random();
      const x1 = Math.random();
      X.push([x0, x1]);
      y.push(x0 + x1 > 1 ? 1 : 0);
    }
    const forest = new RandomForest(5, 5, 2);
    forest.fit(X, y);
    const pred = forest.predict(X);
    expect(accuracy_score(y, pred)).toBeGreaterThan(0.5);
  });
});

describe('KMeans', () => {
  test('fit and predict', () => {
    const X1: number[][] = [];
    const X2: number[][] = [];
    for (let i = 0; i < 30; i++) {
      X1.push([Math.random() + 2, Math.random() + 2]);
      X2.push([Math.random() - 2, Math.random() - 2]);
    }
    const X = [...X1, ...X2];
    const kmeans = new KMeans(2, 300, 5);
    kmeans.fit(X);
    const labels = kmeans.predict(X);
    expect(labels.length).toBe(X.length);
  });
});

describe('PCA', () => {
  test('fit_transform reduces dimensions', () => {
    const X: number[][] = [];
    for (let i = 0; i < 100; i++) {
      X.push([Math.random(), Math.random(), Math.random(), Math.random()]);
    }
    const pca = new PCA(2);
    const Xt = pca.fit_transform(X);
    expect(Xt.length).toBe(100);
    expect(Xt[0].length).toBe(2);
  });
});

describe('StandardScaler', () => {
  test('zero mean per column', () => {
    const X = [[1, 2], [3, 4], [5, 6]];
    const scaler = new StandardScaler();
    const Xs = scaler.fit_transform(X);
    for (let col = 0; col < 2; col++) {
      let mean = 0;
      for (let row = 0; row < Xs.length; row++) mean += Xs[row][col];
      mean /= Xs.length;
      expect(Math.abs(mean)).toBeLessThan(1e-10);
    }
  });
});

describe('train_test_split', () => {
  test('splits with correct ratio', () => {
    const X = Array.from({ length: 100 }, (_, i) => [i]);
    const y = Array.from({ length: 100 }, (_, i) => i);
    const [X_train, X_test, y_train, y_test] = train_test_split(X, y, 0.2, null);
    expect(X_train.length).toBe(80);
    expect(X_test.length).toBe(20);
    expect(y_train.length).toBe(80);
    expect(y_test.length).toBe(20);
  });
});

describe('Metrics', () => {
  test('accuracy_score', () => {
    expect(accuracy_score([0, 1, 0, 1], [0, 1, 0, 0])).toBe(0.75);
  });

  test('mean_squared_error', () => {
    const mse = mean_squared_error([1, 2, 3, 4], [1.1, 2.1, 2.9, 4.1]);
    expect(mse).toBeLessThan(0.1);
  });

  test('r2_score', () => {
    const r2 = r2_score([1, 2, 3, 4, 5], [1.1, 2.1, 2.9, 4.1, 4.9]);
    expect(r2).toBeGreaterThan(0.95);
  });
});