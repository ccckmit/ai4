"""Example usage of the ml package."""

import numpy as np
from ml import (
    LinearRegression, LogisticRegression, DecisionTree,
    RandomForest, KMeans, PCA,
    StandardScaler, train_test_split,
    accuracy_score, mean_squared_error
)

np.random.seed(42)

print("=== Linear Regression ===")
X = np.random.randn(100, 3)
y = 3 * X[:, 0] + 2 * X[:, 1] - X[:, 2] + 1
model = LinearRegression(lr=0.1, n_iterations=1000)
model.fit(X, y)
pred = model.predict(X)
print(f"R2 Score: {1 - np.sum((y - pred)**2) / np.sum((y - np.mean(y))**2):.4f}")

print("\n=== Logistic Regression ===")
X = np.random.randn(100, 2)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
model = LogisticRegression(lr=0.1, n_iterations=500)
model.fit(X, y)
pred = model.predict(X)
print(f"Accuracy: {accuracy_score(y, pred):.4f}")

print("\n=== Decision Tree ===")
X = np.array([[1], [2], [3], [4], [5], [6]], dtype=float)
y = np.array([0, 0, 0, 1, 1, 1])
tree = DecisionTree(max_depth=3)
tree.fit(X, y)
print(f"Predictions: {tree.predict(X)}")

print("\n=== Random Forest ===")
X = np.random.randn(100, 4)
y = (X[:, 0] + X[:, 1] > 0).astype(int)
forest = RandomForest(n_estimators=5, max_depth=5)
forest.fit(X, y)
print(f"Accuracy: {accuracy_score(y, forest.predict(X)):.4f}")

print("\n=== KMeans Clustering ===")
X1 = np.random.randn(30, 2) + [2, 2]
X2 = np.random.randn(30, 2) + [-2, -2]
X = np.vstack([X1, X2])
kmeans = KMeans(n_clusters=2, n_init=5)
kmeans.fit(X)
print(f"Cluster sizes: {np.bincount(kmeans.predict(X))}")

print("\n=== PCA ===")
X = np.random.randn(100, 5)
pca = PCA(n_components=2)
X_transformed = pca.fit_transform(X)
print(f"Original shape: {X.shape} -> Transformed shape: {X_transformed.shape}")

print("\n=== StandardScaler ===")
X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(f"Mean after scaling: {X_scaled.mean(axis=0)}")
print(f"Std after scaling: {X_scaled.std(axis=0)}")

print("\nAll examples completed successfully!")