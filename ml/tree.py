"""Decision Tree classifier and regressor."""

import numpy as np


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
        self.is_classification = None

    def fit(self, X, y):
        y_int = y.astype(int)
        self.is_classification = len(np.unique(y_int)) < len(y)
        self.n_classes = len(np.unique(y_int))
        self.tree = self._build_tree(X, y, depth=0)

    def _best_split(self, X, y):
        best_gain = -1
        best_split = None
        n_samples, n_features = X.shape

        for feat_idx in range(n_features):
            thresholds = np.unique(X[:, feat_idx])
            if len(thresholds) < 2:
                continue
            for t in thresholds[1:]:
                left_mask = X[:, feat_idx] <= t
                right_mask = ~left_mask
                if np.sum(left_mask) < self.min_samples_split:
                    continue
                gain = self._information_gain(y, left_mask, right_mask)
                if gain > best_gain:
                    best_gain = gain
                    best_split = {"feat_idx": feat_idx, "threshold": t}

        return best_split

    def _information_gain(self, y, left_mask, right_mask):
        parent_entropy = self._entropy(y)
        n = len(y)
        n_l, n_r = np.sum(left_mask), np.sum(right_mask)
        if n_l == 0 or n_r == 0:
            return 0
        e_l = self._entropy(y[left_mask])
        e_r = self._entropy(y[right_mask])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r
        return parent_entropy - child_entropy

    def _entropy(self, y):
        if len(y) == 0:
            return 0
        probs = np.bincount(y.astype(int)) / len(y)
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs))

    def _build_tree(self, X, y, depth):
        if depth >= self.max_depth or len(y) < 2 * self.min_samples_split:
            return self._leaf_value(y)

        split = self._best_split(X, y)
        if split is None:
            return self._leaf_value(y)

        left_mask = X[:, split["feat_idx"]] <= split["threshold"]
        right_mask = ~left_mask
        if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
            return self._leaf_value(y)

        return {
            "feat_idx": split["feat_idx"],
            "threshold": split["threshold"],
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[right_mask], y[right_mask], depth + 1),
        }

    def _leaf_value(self, y):
        if len(y) == 0:
            return {"leaf": True, "value": 0}
        if self.is_classification:
            return {"leaf": True, "value": int(np.bincount(y.astype(int)).argmax())}
        return {"leaf": True, "value": float(np.mean(y))}

    def predict(self, X):
        return np.array([self._traverse(x, self.tree) for x in X])

    def _traverse(self, x, node):
        if node.get("leaf"):
            return node["value"]
        if x[node["feat_idx"]] <= node["threshold"]:
            return self._traverse(x, node["left"])
        return self._traverse(x, node["right"])