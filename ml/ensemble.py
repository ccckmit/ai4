"""Ensemble methods: RandomForest."""

from __future__ import annotations

import numpy as np
from .tree import DecisionTree


class RandomForest:
    def __init__(
        self,
        n_estimators: int = 10,
        max_depth: int = 10,
        min_samples_split: int = 2,
    ) -> None:
        self.n_estimators: int = n_estimators
        self.max_depth: int = max_depth
        self.min_samples_split: int = min_samples_split
        self.trees: list[DecisionTree] = []
        self.is_classification: bool = True
        self.n_classes: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.is_classification = len(np.unique(y)) < len(y) / 2
        self.n_classes = len(np.unique(y))
        self.trees = []
        n_samples = X.shape[0]

        for _ in range(self.n_estimators):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.is_classification:
            majority_votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, predictions)
            return majority_votes
        return np.mean(predictions, axis=0)