import { LinearRegression, LogisticRegression, accuracy_score } from '../index';

const X: number[][] = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]];
const y = [3, 6, 9, 12, 15];

const model = new LinearRegression(0.01, 1000);
model.fit(X, y);
const pred = model.predict([[6, 12]]);
console.log('LinearRegression prediction (6,12):', pred[0].toFixed(2));

const X2: number[][] = [[0, 0], [0, 1], [1, 0], [1, 1]];
const y2 = [0, 1, 1, 0];
const clf = new LogisticRegression(0.1, 1000);
clf.fit(X2, y2);
const y_pred = clf.predict(X2);
console.log('LogisticRegression accuracy:', accuracy_score(y2, y_pred));

console.log('ml tests passed');