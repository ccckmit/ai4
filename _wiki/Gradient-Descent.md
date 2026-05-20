# Gradient Descent（梯度下降）

梯度下降是最佳化領域最基礎也最重要的演算法，用於找到函數的局部或全域最小值。其核心思想是：沿著函數梯度的反方向（最陡下降方向）逐步調整變數，逐步逼近極小值點。

## 基本原理

設目標函數 $f(\theta)$，我們想找到 $\theta^*$ 使 $f(\theta^*) = \min_\theta f(\theta)$。

若梯度 $\nabla f(\theta)$ 存在，根據泰勒展開：
$$f(\theta + \Delta\theta) \approx f(\theta) + \nabla f(\theta)^T \Delta\theta + \frac{1}{2}\Delta\theta^T H \Delta\theta + ...$$

要讓 $f$ 下降，需選擇 $\Delta\theta$ 使 $\nabla f(\theta)^T \Delta\theta < 0$。當 $\Delta\theta = -\alpha \nabla f(\theta)$（α > 0）時，
$$\nabla f(\theta)^T (-\alpha \nabla f(\theta)) = -\alpha \|\nabla f(\theta)\|^2 \leq 0$$

保證下降（除非梯度已為零）。這就是梯度下降更新規則：
$$\theta \leftarrow \theta - \alpha \nabla_\theta f(\theta)$$

其中 α 為**學習率（learning rate）**，控制每步邁出的幅度。

## 幾何直覺

梯度是一個向量，指向函數值增加最快的方向。因此負梯度方向就是函數值下降最快的方向。梯度下降就像從山坡上的某一點出發，每次選擇最陡的下坡方向邁出一步，反覆直到到達山谷（局部極小值）。

學習率的作用：
- **太大**：可能越過極小值點甚至發散
- **太小**：收斂緩慢，計算成本高
- **適中**：穩定收斂

## 收斂性

對於**凸函數（convex function）**，梯度下降在滿足一定條件下保證收斂到全域極小值。對於**非凸函數**（如深度學習 Loss 曲面），只能保證收斂到局部極小值或鞍點。

非凸優化的收斂速率取決於函數的光滑性（smoothness）和梯度 Lipschitz 常數 L：

$$| \nabla f(x) - \nabla f(y) | \leq L | x - y |$$

對於 L-smooth 函數，標準梯度下降的收斂速率為 $O(1/k)$（k 為迭代次數），即經過 k 步後誤差為 O(1/k)。

## 變體分類

### 批量梯度下降（Batch Gradient Descent）
每次使用**全部訓練資料**計算梯度後更新參數。優點是梯度估計精確，收斂穩定；缺點是每步計算代價大，記憶體需求高，無法處理百萬筆資料。

$$\theta \leftarrow \theta - \alpha \frac{1}{N} \sum_{i=1}^N \nabla_\theta L_i(\theta)$$

### 隨機梯度下降（SGD）
每次只使用**一個樣本**估算梯度。高頻率更新、收斂快，但梯度估計噪聲大，可能在極小值附近震盪。通常會配合學習率衰減（learning rate decay）來減小震盪。

$$\theta \leftarrow \theta - \alpha \nabla_\theta L_i(\theta) \quad \text{(random } i\text{)}$$

### 小批量梯度下降（Mini-batch GD）
每次使用 **b 個樣本**（mini-batch）。這是深度學習的事實標準（稱為 SGD），結合了上述兩種方法的優點：
- 每步計算量可控（約 $b \times \text{forward\_pass}$）
- 梯度估計足夠穩定
- 矩陣運算可高效並行

典型 batch size：32、64、128、256。較大的 batch size 每次更新更穩定但泛化能力可能略差（sharp minimum）；較小的 batch size 訓練更不穩定但傾向找到 flat minimum，泛化更好。

## 動量法（Momentum）

標準 SGD 的問題是在峽谷（ravine）狀地形中——一個方向陡峭另一個方向平緩——會在陡峭方向持續震盪。

動量法引入速度項累積歷史梯度方向：

$$v \leftarrow \beta v + (1-\beta) \nabla_\theta f(\theta)$$
$$\theta \leftarrow \theta - \alpha v$$

直覺上：梯度在相同方向上累積，在震盪方向相互抵消。這讓收斂更快、更穩定。β 常設為 0.9。

## 自適應學習率方法

傳統 SGD 使用固定或逐步衰減的學習率。自適應方法為每個參數維度單獨維護學習率：

### AdaGrad
對稀疏特徵友好，但學習率會持續衰減，可能過早停止學習。
$$r \leftarrow r + \nabla_\theta f(\theta) \odot \nabla_\theta f(\theta)$$
$$\theta \leftarrow \theta - \frac{\alpha}{\sqrt{r + \epsilon}} \odot \nabla_\theta f(\theta)$$

### RMSProp
指數加權移動平均，解決 AdaGrad 學習率衰減過快的問題。Hinton 建議 β=0.9, α=0.001。

### Adam（Adaptive Moment Estimation）
目前最流行的優化器，結合動量法和 RMSProp：

$$m \leftarrow \beta_1 m + (1-\beta_1) g \quad \text{（動量，梯度的指數加權移動平均）}$$
$$v \leftarrow \beta_2 v + (1-\beta_2) g^2 \quad \text{（二階矩估計）}$$
$$\hat{m} \leftarrow \frac{m}{1-\beta_1^t} \quad \text{（偏差校正）}$$
$$\hat{v} \leftarrow \frac{v}{1-\beta_2^t}$$
$$\theta \leftarrow \theta - \alpha \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}$$

默認參數：β₁=0.9, β₂=0.999, ε=10⁻⁸。Adam 在多數任務上表現良好，是默認優化器的首選。

## 學習率排程

固定的學習率可能不是最優。常用排程策略：
- **Step Decay**：每 N 個 epoch 衰減一次（如減半）
- **Cosine Annealing**：按餘弦函數週期性衰減
- **Warmup**：開始時逐漸增大學習率，再衰減（避免早期不穩定）
- **Reduce on Plateau**：驗證 loss 停滯時自動減小學習率

## 本專案中的實現

本專案 `nn/optim.py` 實現了 Adam 優化器（`Adam` 類），用於訓練 GPT 模型。它維護兩個指數加權移動平均（m 和 v）並進行偏差校正，確保在訓練初期仍有合理的學習率。訓練迴圈（在 `chargpt.py` 中）固定迭代次數，無需 learning rate schedule，但在實務上可視需求加入 warmup 或衰減機制。

---

**上一篇**：[Backpropagation.md](Backpropagation.md)

**相關連結**：[Adam.md](Adam.md) | [Q-Learning.md](Q-Learning.md)