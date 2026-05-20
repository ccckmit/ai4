export function accuracy_score(y_true: number[], y_pred: number[]): number {
  let correct = 0;
  for (let i = 0; i < y_true.length; i++) if (y_true[i] === y_pred[i]) correct++;
  return correct / y_true.length;
}

export function mean_squared_error(y_true: number[], y_pred: number[]): number {
  return y_true.reduce((s, y, i) => s + (y - y_pred[i]) ** 2, 0) / y_true.length;
}

export function r2_score(y_true: number[], y_pred: number[]): number {
  const mean = y_true.reduce((s, v) => s + v, 0) / y_true.length;
  const ss_res = y_true.reduce((s, y, i) => s + (y - y_pred[i]) ** 2, 0);
  const ss_tot = y_true.reduce((s, y) => s + (y - mean) ** 2, 0);
  return 1 - ss_res / ss_tot;
}

export function confusion_matrix(y_true: number[], y_pred: number[]): number[][] {
  const classes = [...new Set([...y_true, ...y_pred])].sort((a, b) => a - b);
  const n = classes.length;
  const cm = Array.from({ length: n }, () => new Array(n).fill(0));
  for (let i = 0; i < y_true.length; i++) {
    const ti = classes.indexOf(y_true[i]);
    const pi = classes.indexOf(y_pred[i]);
    if (ti >= 0 && pi >= 0) cm[ti][pi]++;
  }
  return cm;
}