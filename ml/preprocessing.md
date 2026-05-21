# ml/preprocessing.md - 資料預處理理論

本模組實現了機器學習流程中的資料預處理工具：**StandardScaler**、**MinMaxScaler** 和 **train_test_split**。預處理是確保模型穩定訓練和公平評估的關鍵步驟。

## 標準化（StandardScaler）

### Z-Score 標準化

將資料轉換為均值為 0、標準差為 1 的分布：

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

- $\mu$：訓練集的均值
- $\sigma$：訓練集的標準差

### 為什麼需要標準化

1. **尺度統一**：不同特徵可能量級差異巨大（如年齡 0-100 vs 收入 0-10⁷）
2. **梯度下降加速**：特徵尺度均勻時，梯度下降更穩定、收斂更快
3. **距離度量公平**：KNN、K-Means、SVM 等依賴距離的演算法，若不標準化，量級大的特徵會主導距離計算
4. **正則化效果**：L1/L2 正則化假設所有特徵尺度相同

### 擬合（fit）與轉換（transform）

標準化的統計量（$\mu, \sigma$）必須只從**訓練集**計算，然後用同樣的統計量轉換訓練集和測試集：

```python
scaler = StandardScaler()
scaler.fit(X_train)               # 計算 μ, σ
X_train_scaled = scaler.transform(X_train)
X_test_scaled  = scaler.transform(X_test)  # 使用訓練集的 μ, σ
```

這防止了**資料洩漏（data leakage）**——測試集的資訊不應影響訓練過程。

## 最小-最大正規化（MinMaxScaler）

將資料縮放到 $[0, 1]$ 區間：

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

- 對離群值敏感（一個極端值會壓縮其他資料的分布）
- 適合需要固定輸出範圍的場景（如神經網路的輸入）

### StandardScaler vs MinMaxScaler

| 特性 | StandardScaler | MinMaxScaler |
|------|---------------|--------------|
| 輸出範圍 | 無界（通常 $[-3, 3]$） | $[0, 1]$ |
| 離群值 | 較穩健 | 敏感 |
| 分布假設 | 近似常態分布 | 無 |
| 適用場景 | 線性模型、距離模型 | 神經網路、影像處理 |

## 訓練測試分割（train_test_split）

將資料隨機分為訓練集和測試集：

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### 為什麼需要分割

- **評估泛化能力**：測試集模擬未見過的資料
- **防止過擬合**：如果只在訓練集上評估，模型可能只是記住了資料
- **超參數選擇**：可再分出驗證集做模型選擇

### 實現細節

1. 生成隨機排列的索引（shuffle）
2. 按 `test_size` 比例分割
3. 確保 X 和 y 的對應關係不變

典型比例：訓練 80%、測試 20%，或訓練 70%、驗證 15%、測試 15%。

---

**相關連結**：[linear_models.md](linear_models.md) | [metrics.md](metrics.md)
