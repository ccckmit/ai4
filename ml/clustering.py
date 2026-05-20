"""Clustering algorithms: KMeans."""

from __future__ import annotations

import numpy as np


class KMeans:
    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        n_init: int = 10,
    ) -> None:
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.n_init: int = n_init
        self.centroids: np.ndarray | None = None
        self.labels: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> KMeans:
        best_inertia: float | None = None
        best_centroids: np.ndarray | None = None
        best_labels: np.ndarray | None = None

        for _ in range(self.n_init):
            indices = np.random.choice(len(X), self.n_clusters, replace=False)
            centroids = X[indices].copy()

            for _ in range(self.max_iter):
                labels = self._assign_labels(X, centroids)
                new_centroids = self._compute_centroids(X, labels)
                if np.allclose(centroids, new_centroids):
                    break
                centroids = new_centroids

            inertia = self._compute_inertia(X, labels, centroids)
            if best_inertia is None or inertia < best_inertia:
                best_inertia = inertia
                best_centroids = centroids
                best_labels = labels

        self.centroids = best_centroids
        self.labels = best_labels
        return self

    def _assign_labels(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _compute_inertia(self, X: np.ndarray, labels: np.ndarray, centroids: np.ndarray) -> float:
        return np.sum((X - centroids[labels]) ** 2)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._assign_labels(X, self.centroids)