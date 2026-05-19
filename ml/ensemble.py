"""Ensemble methods: RandomForest."""

import numpy as np
from .tree import DecisionTree


class RandomForest:
    def __init__(self, n_estimators=10, max_depth=10, min_samples_split=2):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = []
        self.is_classification = True

    def fit(self, X, y):
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

    def predict(self, X):
        predictions = np.array([tree.predict(X) for tree in self.trees])
        if self.is_classification:
            majority_votes = np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, predictions)
            return majority_votes
        return np.mean(predictions, axis=0)