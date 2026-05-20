"""Dimensionality reduction: PCA."""

from __future__ import annotations

import numpy as np


class PCA:
    def __init__(self, n_components: int | None = None) -> None:
        self.n_components: int | None = n_components
        self.components: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.explained_variance: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> PCA:
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        cov = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        if self.n_components is not None:
            eigenvectors = eigenvectors[:, : self.n_components]

        self.components = eigenvectors.T
        self.explained_variance = eigenvalues[: self.n_components]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        return np.dot(X_transformed, self.components) + self.mean