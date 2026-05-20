import { LinearRegression, LogisticRegression, accuracy_score, train_test_split, StandardScaler } from '../index';

const X = [[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]];
const y = [3, 6, 9, 12, 15];

const model = new LinearRegression(0.01, 1000);
model.fit(X, y);

const pred = model.predict([[6, 12]]);
console.log('LinearRegression prediction (6,12):', pred);

const X2: number[][] = [[0, 0], [0, 1], [1, 0], [1, 1], [0, 0], [0, 1], [1, 0], [1, 1]];
const y2 = [0, 1, 1, 0, 0, 1, 1, 0];

const { X_train, X_test, y_train, y_test } = train_test_split(X2, y2, 0.25, 42);
const clf = new LogisticRegression(0.1, 1000);
clf.fit(X_train, y_train);
const y_pred = clf.predict(X_test);
console.log('LogisticRegression accuracy:', accuracy_score(y_test, y_pred));

const scaler = new StandardScaler();
const X_scaled = scaler.fit_transform(X);
console.log('StandardScaler mean:', scaler.mean, 'std:', scaler.std);