# Linear Regression（線性迴歸）

線性迴歸 (linear regression) 是統計學和機器學習中最基礎的回歸模型，用於建立特徵 (features) 與連續目標變數 (continuous target variable) 之間的線性關係。其核心假設為目標變數是特徵的線性組合加上隨機誤差。

## 模型定義

給定資料集 $\{(x_i, y_i)\}_{i=1}^N$，其中 $x_i \in \mathbb{R}^d$ 為特徵向量，$y_i \in \mathbb{R}$ 為目標值。線性迴歸模型假設：

$$y = w_1 x_1 + w_2 x_2 + \cdots + w_d x_d + b + \varepsilon$$

向量形式：

$$\hat{y} = w^T x + b$$

其中 $w \in \mathbb{R}^d$ 為權重向量 (weight vector)，$b \in \mathbb{R}$ 為偏置項 (bias/intercept)，$\varepsilon$ 為隨機誤差項。

為了方便矩陣運算，通常將偏置項 $b$ 合併進權重向量：在特徵中加入常數 1，形成增廣特徵向量 $\tilde{x} = [1, x_1, ..., x_d]^T$ 與增廣權重 $\tilde{w} = [b, w_1, ..., w_d]^T$，則：

$$\hat{y} = \tilde{w}^T \tilde{x}$$

對全部樣本使用設計矩陣 (design matrix) $X \in \mathbb{R}^{N \times (d+1)}$：

$$\hat{y} = X \tilde{w}$$

## 假設（Hypothesis）

線性迴歸的統計推斷依賴以下關鍵假設（Gauss-Markov 定理）：

### 1. 線性性（Linearity）
目標變數 $y$ 與參數 $w$ 之間存在線性關係。注意這是指對**參數**的線性，而非對特徵的線性。$y = w_1 x + w_2 x^2$ 仍然是線性模型（對參數 $w_1, w_2$ 線性）。

### 2. 獨立性（Independence）
樣本之間相互獨立。違反此假設的情況包括時間序列資料（自相關）。

### 3. 同方差性（Homoscedasticity）
誤差項 $\varepsilon$ 的變異數為常數：$\text{Var}(\varepsilon_i) = \sigma^2$，不隨 $x_i$ 變化。若變異數非常數，稱為異方差性 (heteroscedasticity)。

### 4. 常態性（Normality）
誤差項服從常態分佈：$\varepsilon \sim \mathcal{N}(0, \sigma^2)$。此假設對於最小平方法估計的一致性非必要，但對信賴區間和假設檢定很重要。

### 5. 無多重共線性（No Multicollinearity）
特徵之間不存在精確的線性相關。若存在完全共線性，$(X^T X)$ 不可逆，正規方程式無法求解。

## 成本函數（Cost Function）

最常見的成本函數為均方誤差 (Mean Squared Error, MSE)，也稱為殘差平方和 (Residual Sum of Squares, RSS)：

$$J(w, b) = \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)^2 = \frac{1}{N} \|X\tilde{w} - y\|^2$$

最小化 MSE 等價於最大概似估計 (Maximum Likelihood Estimation, MLE)，假設誤差服從常態分佈。

## 封閉解：正規方程式（Normal Equation）

對成本函數求梯度並設為零：

$$\nabla_{\tilde{w}} J = \frac{2}{N} X^T (X\tilde{w} - y) = 0$$

得到正規方程式的封閉解 (closed-form solution)：

$$\tilde{w} = (X^T X)^{-1} X^T y$$

### 計算複雜度
- 計算 $(X^T X)^{-1}$ 需要 $O(d^3 + N d^2)$ 時間
- 當 $d$（特徵數）很大時不可行
- $(X^T X)$ 可能為奇異矩陣（奇異或病態），此時可使用偽逆矩陣 (pseudo-inverse) $X^+ = (X^T X)^{-1} X^T$ 或正則化

### 與梯度下降的比較

| 方法 | 優點 | 缺點 |
|------|------|------|
| 正規方程式 | 一次計算得全局最優解 | $O(d^3)$ 計算，記憶體 $O(Nd)$ |
| 梯度下降 | 可擴展至大資料，$O(kNd)$ k 為迭代次數 | 需調整學習率，不收斂於非凸問題 |

## 梯度下降求解

梯度下降 (gradient descent) 迭代更新參數：

$$w \leftarrow w - \alpha \cdot \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i) x_i$$
$$b \leftarrow b - \alpha \cdot \frac{1}{N} \sum_{i=1}^N (\hat{y}_i - y_i)$$

其中 $\alpha$ 為學習率 (learning rate)。

本專案 `ml/linear_models.py:LinearRegression` 使用梯度下降實現，初始化權重為零向量，迭代更新直至收斂或達到最大迭代次數。

```python
# ml/linear_models.py (簡化)
for _ in range(self.n_iterations):
    y_pred = np.dot(X, self.weights) + self.bias
    dw = (1 / n_samples) * np.dot(X.T, y_pred - y)
    db = (1 / n_samples) * np.sum(y_pred - y)
    self.weights -= self.lr * dw
    self.bias -= self.lr * db
```

## 係數解釋（Coefficient Interpretation）

在標準線性迴歸中，權重 $w_j$ 解釋為：「在其他特徵不變的情況下，$x_j$ 每增加一個單位，$\hat{y}$ 平均增加 $w_j$ 個單位。」

這稱為邊際效應 (marginal effect)。需要注意的是：

- 若特徵之間相關，係數解釋會變得複雜
- 標準化後的係數可比較特徵重要性
- 對數轉換後的變數需按彈性 (elasticity) 解釋

## 評估指標

### R² 判定係數（Coefficient of Determination）

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_i (y_i - \hat{y}_i)^2}{\sum_i (y_i - \bar{y})^2}$$

- $SS_{res}$（殘差平方和）：模型未解釋的變異
- $SS_{tot}$（總平方和）：目標的總變異
- $R^2 \in [0, 1]$，越大表示模型擬合越好

**R² 的解讀**：目標變數中可由特徵解釋的變異比例。例如 $R^2 = 0.8$ 表示 80% 的目標變異可由模型解釋。

**注意事項**：
- 加入更多特徵時 $R^2$ 永不下降（即使特徵無用），因此需要調整 R² (Adjusted R²)：
  $$\bar{R}^2 = 1 - \frac{SS_{res}/(N-d-1)}{SS_{tot}/(N-1)}$$
- $R^2$ 在非線性關係上可能誤導
- $R^2$ 不能證明因果關係

### 其他指標
- **均方誤差 (MSE)**：$\frac{1}{N}\sum(y_i - \hat{y}_i)^2$
- **均方根誤差 (RMSE)**：$\sqrt{MSE}$，與 $y$ 同單位
- **平均絕對誤差 (MAE)**：$\frac{1}{N}\sum|y_i - \hat{y}_i|$

本專案 `ml/metrics.py` 實現了 `r2_score` 和 `mean_squared_error` 兩個評估函數：

```python
def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
```

## 多項式特徵擴展（Polynomial Feature Extension）

線性迴歸的「線性」是指對**參數**線性，而非對特徵。因此可以透過加入特徵的多項式組合來擬合非線性關係：

$$\hat{y} = w_0 + w_1 x + w_2 x^2 + w_3 x^3 + \cdots + w_p x^p$$

這稱為多項式迴歸 (polynomial regression)，仍可用正規方程式求解。

```mermaid
graph LR
    subgraph 原始特徵
        x1
        x2
    end
    subgraph 多項式擴展
        x1
        x2
        x1^2
        x2^2
        x1*x2
    end
    subgraph 線性模型
        w0 + w1*x1 + w2*x2 + w3*x1^2 + w4*x2^2 + w5*x1*x2
    end
```

### 多項式擴展的陷阱

- **過擬合 (overfitting)**：高次多項式容易過度擬合訓練資料
- **特徵膨脹**：$d$ 個原始特徵、$p$ 次多項式會產生 $O(d^p)$ 個新特徵
- **數值不穩定**：高次項導致數值範圍差異巨大，需搭配特徵標準化

維度災難的一種表現：隨著多項式次數增加，參數空間急遽增大。

## 正則化（Regularization）

當特徵數量大或存在多重共線性時，可在成本函數中加入正則化項：

- **Ridge Regression (L2)**：$J = MSE + \lambda \sum w_j^2$
- **Lasso Regression (L1)**：$J = MSE + \lambda \sum |w_j|$
- **Elastic Net**：Combination of L1 and L2

正則化在縮小權重值的同時，可減少模型變異數、改善泛化能力。

## 線性迴歸的局限性

1. **僅捕捉線性關係**：若真實關係為高度非線性，表現不佳（可透過多項式特徵緩解）
2. **對離群值敏感**：MSE 對大誤差賦予平方權重，離群值會嚴重影響擬合
3. **同方差性假設**：異方差資料會導致不可靠的推論
4. **多重共線性**：導致係數估計不穩定，標準誤過大
5. **外推能力有限**：在訓練資料範圍外的預測可能完全失準

---

**上一篇**：無（此為系列新條目）

**相關連結**：[Logistic-Regression.md](Logistic-Regression.md) | [StandardScaler.md](StandardScaler.md) | [Gradient-Descent.md](Gradient-Descent.md) | [Train-Test-Split.md](Train-Test-Split.md)
