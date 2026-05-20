# ml/linear_models.md - 線性模型理論

本模組實現了兩種最基礎的監督學習模型：**線性回歸**（Regression）和**邏輯斯回歸**（Classification）。兩者都是機器學習的基石，概念簡單但強大，是理解更複雜模型的起點。

## 線性回歸（Linear Regression）

### 問題設定

給定 n 筆資料 $(x_1, y_1), ..., (x_n, y_n)$，其中 $x_i \in \mathbb{R}^d$（特徵），$y_i \in \mathbb{R}$（連續目標），線性回歸假設：

$$y = w^T x + b + \epsilon$$

其中 $w$ 是權重向量，$b$ 是偏置，$\epsilon$ 是噪聲。

### 目標函數

使用均方誤差（Mean Squared Error, MSE）作為損失：

$$\mathcal{L}(w, b) = \frac{1}{n} \sum_{i=1}^n (w^T x_i + b - y_i)^2$$

我們要找到最小化這個損失的 $w$ 和 $b$。

### 梯度推導

對 MSE 求導：

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{2}{n} \sum_{i=1}^n (w^T x_i + b - y_i) x_i = \frac{2}{n} X^T (Xw + b - y)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^n (w^T x_i + b - y_i) = \frac{2}{n} \mathbf{1}^T (Xw + b - y)$$

### 解析解

令 $\hat{X} = [X, \mathbf{1}]$（加上一列全 1），$\hat{w} = [w; b]$，則：

$$\hat{w}^* = (\hat{X}^T \hat{X})^{-1} \hat{X}^T y$$

這稱為**正規方程式（Normal Equation）**。但當特徵維度高或樣本多時，矩陣求逆 O(d³) 代價高，通常用梯度下降。

### 本專案實現

```python
for _ in range(self.n_iterations):
    y_pred = X @ self.weights + self.bias
    dw = (1/n) * X.T @ (y_pred - y)
    db = (1/n) * sum(y_pred - y)
    self.weights -= self.lr * dw
    self.bias -= self.lr * db
```

這是標準的批量梯度下降，每次迭代遍歷全部樣本。

## 邏輯斯回歸（Logistic Regression）

### 為什麼需要非線性

線性回歸輸出連續值，無法直接用於分類（輸出類別機率）。需要一個將線性輸出轉換為 [0,1] 區間的函數——**Sigmoid**。

### Sigmoid 函數

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

- z → +∞ 時，σ(z) → 1
- z → -∞ 時，σ(z) → 0
- σ(0) = 0.5

輸出可以解釋為「類別 1 的機率」。

### 目標函數：二元交叉熵

對於二元分類，使用交叉熵損失：

$$\mathcal{L} = -\frac{1}{n} \sum_{i=1}^n [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$$

其中 $\hat{y}_i = \sigma(w^T x_i + b)$。

這個損失函數的優點：
- 當預測錯誤時，梯度較大，學習快
- 當預測正確時，梯度趨近於零，震盪小

### 梯度推導

對交叉熵損失求導：

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{1}{n} \sum_i (\sigma(z_i) - y_i) x_i = \frac{1}{n} X^T (\hat{y} - y)$$

形式上與線性回歸相同，差別在於 $\hat{y}$ 是 sigmoid 輸出而非直接線性輸出。

### 預測方式

```python
y_pred = sigmoid(X @ w + b)
return (y_pred >= 0.5).astype(int)  # 閾值 0.5 決策
```

類別 1 的機率 ≥ 0.5 則預測為 1，否則為 0。

### 決策邊界

邏輯斯回歸的決策邊界是**線性的**（$w^T x + b = 0$）。這是名稱中「線性」的由來——不是說模型輸出是線性的，而是決策邊界是線性超平面。

對於非線性可分的資料（如 XOR），邏輯斯回歸無法完美分類，需要：
- 特徵工程（增加多項式特徵）
- 核方法（SVM with kernel trick）
- 神經網路（非線性變換）

## 正則化（概念）

本實現未包含正則化，但實務上常用 L1/L2 正則化避免過擬合：

- **L2 正則化（Ridge）**：在損失函數中加入 $\lambda \|w\|^2$，傾向讓權重趨近於零
- **L1 正則化（Lasso）**：加入 $\lambda \|w\|_1$，傾向產生稀疏權重（特徵選擇）

---

**相關連結**：[tree.md](tree.md) | [ensemble.md](ensemble.md)