"""Tests for ml package."""

import numpy as np
import pytest

from ml.linear_models import LinearRegression, LogisticRegression
from ml.tree import DecisionTree
from ml.ensemble import RandomForest
from ml.clustering import KMeans
from ml.decomposition import PCA
from ml.preprocessing import StandardScaler, train_test_split
from ml.metrics import accuracy_score, mean_squared_error, r2_score, confusion_matrix


class TestLinearRegression:
    def test_fit_predict(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([2, 4, 6, 8, 10], dtype=float)
        model = LinearRegression(lr=0.01, n_iterations=1000)
        model.fit(X, y)
        pred = model.predict(X)
        assert pred.shape == y.shape
        assert np.allclose(pred, y, atol=0.5)

    def test_multi_feature(self):
        X = np.random.randn(100, 3)
        y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 1
        model = LinearRegression(lr=0.1, n_iterations=1000)
        model.fit(X, y)
        pred = model.predict(X)
        assert r2_score(y, pred) > 0.9


class TestLogisticRegression:
    def test_binary_classification(self):
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        model = LogisticRegression(lr=0.1, n_iterations=1000)
        model.fit(X, y)
        pred = model.predict(X)
        assert accuracy_score(y, pred) > 0.7

    def test_predict_proba(self):
        X = np.random.randn(50, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        model = LogisticRegression(lr=0.1, n_iterations=500)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape == (50, 2)
        assert np.allclose(proba[:, 0] + proba[:, 1], 1)


class TestDecisionTree:
    def test_classification(self):
        X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert accuracy_score(y, pred) >= 0.8

    def test_regression(self):
        X = np.array([[1], [2], [3], [4], [5]], dtype=float)
        y = np.array([1.0, 1.0, 3.0, 3.0, 5.0])
        tree = DecisionTree(max_depth=3)
        tree.fit(X, y)
        pred = tree.predict(X)
        assert np.mean(np.abs(pred - y)) < 1.0


class TestRandomForest:
    def test_classification(self):
        X = np.random.randn(100, 4)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        forest = RandomForest(n_estimators=5, max_depth=5)
        forest.fit(X, y)
        pred = forest.predict(X)
        assert accuracy_score(y, pred) > 0.8


class TestKMeans:
    def test_fit_predict(self):
        np.random.seed(42)
        X1 = np.random.randn(30, 2) + [2, 2]
        X2 = np.random.randn(30, 2) + [-2, -2]
        X = np.vstack([X1, X2])
        kmeans = KMeans(n_clusters=2, n_init=5)
        kmeans.fit(X)
        labels = kmeans.predict(X)
        assert len(np.unique(labels)) == 2
        assert len(labels) == len(X)


class TestPCA:
    def test_fit_transform(self):
        X = np.random.randn(100, 5)
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        assert X_transformed.shape == (100, 2)
        assert pca.components is not None

    def test_inverse_transform(self):
        X = np.random.randn(50, 4)
        pca = PCA(n_components=2)
        X_transformed = pca.fit_transform(X)
        X_reconstructed = pca.inverse_transform(X_transformed)
        assert X_reconstructed.shape == X.shape


class TestStandardScaler:
    def test_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-10)
        assert np.allclose(np.std(X_scaled, axis=0), 1, atol=1e-10)


class TestTrainTestSplit:
    def test_split_sizes(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20

    def test_reproducibility(self):
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        X_train1, _, y_train1, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train2, _, y_train2, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        assert np.array_equal(X_train1, X_train2)
        assert np.array_equal(y_train1, y_train2)


class TestMetrics:
    def test_accuracy_score(self):
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0])
        assert accuracy_score(y_true, y_pred) == 0.75

    def test_mean_squared_error(self):
        y_true = np.array([1, 2, 3, 4])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1])
        mse = mean_squared_error(y_true, y_pred)
        assert mse < 0.1

    def test_r2_score(self):
        y_true = np.array([1, 2, 3, 4, 5])
        y_pred = np.array([1.1, 2.1, 2.9, 4.1, 4.9])
        r2 = r2_score(y_true, y_pred)
        assert r2 > 0.95

    def test_confusion_matrix(self):
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        cm = confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)
        assert cm[0, 0] == 2
        assert cm[1, 1] == 2