# ml/ — 機器學習工具箱

純 NumPy 實現的機器學習演算法集合，涵蓋監督學習、非監督學習、資料预处理和評估指標。

## 模組總覽

| 模組 | 說明 |
|------|------|
| `linear_models.py` | LinearRegression, LogisticRegression |
| `tree.py` | DecisionTree（分類與回歸） |
| `ensemble.py` | RandomForest |
| `clustering.py` | KMeans |
| `decomposition.py` | PCA |
| `preprocessing.py` | StandardScaler, train_test_split |
| `metrics.py` | accuracy, MSE, R², confusion_matrix |

## 使用方式

```python
from ml import LinearRegression, LogisticRegression
from ml.metrics import accuracy_score

model = LinearRegression(lr=0.01, n_iterations=1000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

## 範例

```bash
PYTHONPATH=. uv run python ml/examples/example.py
```

## 測試

```bash
uv run pytest ml/tests
```

## 理論背景

- [linear_models.md](linear_models.md) — 線性回歸與邏輯斯回歸
- 決策樹：基於資訊增益（Information Gain）分裂
- Random Forest：多棵決策樹的集成（bagging）
- PCA：主成分分析，通過 SVD 分解找到方差最大方向
- KMeans：迭代式聚類，收斂到局部最優

## 模組結構

```
ml/
  linear_models.py   # 線性/邏輯斯回歸
  tree.py            # 決策樹
  ensemble.py        # 隨機森林
  clustering.py      # K-Means
  decomposition.py   # PCA
  preprocessing.py  # StandardScaler, train_test_split
  metrics.py         # 評估指標
```