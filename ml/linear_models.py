"""Linear models: LinearRegression, LogisticRegression."""

from __future__ import annotations

import numpy as np


class LinearRegression:
    def __init__(self, lr: float = 0.01, n_iterations: int = 1000) -> None:
        self.lr: float = lr
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            y_pred = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.dot(X, self.weights) + self.bias


class LogisticRegression:
    def __init__(self, lr: float = 0.01, n_iterations: int = 1000) -> None:
        self.lr: float = lr
        self.n_iterations: int = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float | None = None

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for _ in range(self.n_iterations):
            linear = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear)
            dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X: np.ndarray) -> np.ndarray:
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear)
        return (y_pred >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        linear = np.dot(X, self.weights) + self.bias
        y_pred = self._sigmoid(linear)
        return np.column_stack([1 - y_pred, y_pred])