# ml/decomposition.md - PCA 降維理論

本模組實現了**主成分分析（Principal Component Analysis, PCA）**，是最廣泛使用的線性降維技術。

## 降維的動機

高維資料面臨「維度詛咒（Curse of Dimensionality）」的問題：
- 資料稀疏，距離度量失效
- 計算成本高
- 容易過擬合

降維的目標是在保留資料主要結構的前提下降低維度。

## PCA 的核心思想

PCA 尋找資料中**變異數最大**的方向（主成分），將資料投影到這些方向上：

1. **第一主成分**：資料投影後變異數最大的方向
2. **第二主成分**：與第一主成分正交，投影變異數次大的方向
3. 以此類推

## 數學推導

### 中心化

先將資料中心化到原點：

$$X_{centered} = X - \mu$$

### 共變異數矩陣

$$C = \frac{1}{n-1} X_{centered}^T X_{centered}$$

### 特徵值分解

對 $C$ 進行特徵值分解：

$$C v_i = \lambda_i v_i$$

- $\lambda_i$：特徵值，對應主成分的變異數
- $v_i$：特徵向量，對應主成分的方向

### 投影

將原始資料投影到前 k 個主成分：

$$X_{reduced} = X_{centered} \cdot V_k$$

其中 $V_k$ 是前 k 個特徵向量組成的矩陣。

### 解釋變異數比例

第 i 個主成分解釋的變異數比例：

$$PVE_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

累積 PVE 用來決定保留多少維度，通常保留 90-95% 的變異數。

## PCA 的應用

- **視覺化**：降到 2D 或 3D 進行資料探索
- **降噪**：丟棄小變異數的方向（通常對應噪聲）
- **加速**：降低後續演算法的計算成本
- **特徵壓縮**：減少儲存空間

## PCA 的假設與限制

1. **線性假設**：只能捕捉線性結構
2. **變異數≠重要性**：變異數大的方向不一定對分類任務重要
3. **標準化必要**：特徵尺度差異大時，PCA 會被量級大的特徵主導。應先做 `StandardScaler`。
4. **離群值敏感**：少數極端值可能扭曲主成分方向

## 本專案實現

使用 NumPy 的 `np.linalg.svd()`（奇異值分解）實現，SVD 比直接分解共變異數矩陣數值更穩定：

```python
U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
V = Vt.T
X_reduced = X_centered @ V[:, :n_components]
```

詳細理論請見 [_wiki/PCA.md](../_wiki/PCA.md)。

---

**相關連結**：[clustering.md](clustering.md) | [preprocessing.md](preprocessing.md)
