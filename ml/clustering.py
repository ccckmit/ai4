"""Clustering algorithms: KMeans."""

import numpy as np


class KMeans:
    def __init__(self, n_clusters=3, max_iter=300, n_init=10):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.centroids = None
        self.labels = None

    def fit(self, X):
        best_inertia = None
        best_centroids = None
        best_labels = None

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

    def _assign_labels(self, X, centroids):
        distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
        return np.argmin(distances, axis=1)

    def _compute_centroids(self, X, labels):
        return np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])

    def _compute_inertia(self, X, labels, centroids):
        return np.sum((X - centroids[labels]) ** 2)

    def predict(self, X):
        return self._assign_labels(X, self.centroids)