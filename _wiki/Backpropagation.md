# Backpropagation（反向傳播）

反向傳播是訓練深度神經網路的核心演算法，全稱為「誤差反向傳播」（error backpropagation）。它是一種高效的計算梯度（gradient）方法，基於鏈式法則（chain rule）从输出层向输入层逐层传递 loss 對每個參數的偏導數。本質上，反向傳播解決的是「如何根據輸出層的誤差，調整網路中每個參數」的問題。

## 為什麼需要反向傳播

考慮一個神經網路 Loss 函數 $L(\theta)$，其中 $\theta$ 包含所有權重參數。要最小化 L，通常使用梯度下降（gradient descent）：

$$\theta \leftarrow \theta - \alpha \nabla_\theta L(\theta)$$

這需要計算 $\nabla_\theta L(\theta)$——L 對每個參數的偏導數。一個現代網路可能有數百萬個參數，直接計算每個偏導數的代價極高。反向傳播利用計算圖（computation graph）的結構，將複雜函數分解為基本運算的組合，透過鏈式法則高效地重複利用中間計算結果，將計算複雜度從 O(參數數量) 降到與計算一次前向傳播相當。

## 鏈式法則（Chain Rule）

反向傳播的數學基礎是多變數微積分的鏈式法則。對於複合函數：

$$y = f(g(x)) = f(g(x))$$

$$\frac{dy}{dx} = \frac{dy}{dg} \cdot \frac{dg}{dx}$$

推廣到多變數情境：若 $z = f(x_1, x_2, ..., x_n)$，每個 $x_i = g_i(y_1, y_2, ..., y_m)$，則：

$$\frac{\partial z}{\partial y_j} = \sum_i \frac{\partial z}{\partial x_i} \cdot \frac{\partial x_i}{\partial y_j}$$

這正是計算圖中每個節點需要做的：收集上游梯度，乘以本地梯度，傳給下游。

## 前向傳播與計算圖

在前向傳播中，輸入資料依序通過網路每層，最終產生輸出 loss。這個過程中每個中間結果（activations、pre-activations）都被記錄下來，形成一個有向無環計算圖（Directed Acyclic Graph, DAG）。

例如，一個簡單的 2 層網路：
- $z_1 = W_1 x + b_1$（線性變換）
- $a_1 = \sigma(z_1)$（非線性激活，如 ReLU）
- $z_2 = W_2 a_1 + b_2$（線性變換）
- $L = \text{loss}(z_2, y)$（損失函數）

反向傳播時，我們從 loss 開始，先計算 $\partial L / \partial z_2$，然後依次往回傳遞：$\partial L / \partial W_2$、$\partial L / \partial a_1$、$\partial L / \partial z_1$、$\partial L / \partial W_1$、$\partial L / \partial b_1$ 等。

## 反向傳播的步驟

### 1. 前向傳播（Forward Pass）
輸入經過網路，記錄每層的中間變數。用於後續梯度計算。

### 2. 計算輸出層梯度
計算 loss 對輸出層局部梯度。例如均方誤差（MSE）：
$$\frac{\partial L}{\partial \hat{y}} = 2(\hat{y} - y)$$

### 3. 反向遍歷計算圖（Backward Pass）
從輸出層往輸入層方向：
1. 接收上游傳來的梯度 $\frac{\partial L}{\partial \text{out}}$
2. 計算本地每個輸入的梯度：$\frac{\partial \text{out}}{\partial \text{input}}$
3. 將梯度乘積後傳給下游：$\frac{\partial L}{\partial \text{input}} = \frac{\partial L}{\partial \text{out}} \cdot \frac{\partial \text{out}}{\partial \text{input}}$

## 常見運算的梯度

### 矩陣乘法
若 $y = Wx + b$，其中 $W \in \mathbb{R}^{m \times n}$、$x \in \mathbb{R}^n$、$y \in \mathbb{R}^m$：
$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial y} \cdot x^T$$
$$\frac{\partial L}{\partial x} = W^T \cdot \frac{\partial L}{\partial y}$$

### ReLU 激活函數
$$f(x) = \max(0, x)$$
$$\frac{\partial f}{\partial x} = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

### Softmax 函數
若 $p_i = \text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$，則：
$$\frac{\partial p_i}{\partial z_j} = p_i (\delta_{ij} - p_j)$$
其中 $\delta_{ij}$ 為 Kronecker delta。

## 自動微分（Automatic Differentiation）

現代深度學習框架（PyTorch、TensorFlow、JAX）使用**反向模式自動微分**（reverse-mode autodiff）來實現反向傳播。其核心思想是：

1. 將任意複雜的數學運算分解為基本運算的有向無環圖
2. 前向計算時記錄運算順序和中间值
3. 反向傳播時依據鏈式法則自動計算每個節點的梯度

本專案 `nn/tensor.py` 中的 `Tensor` 類就是一個簡化的自動微分框架：每個張量儲存 `data`（數值）、`grad`（梯度）和 `_backward`（梯度計算函數），透過建構計算圖來實現反向傳播。

## 廣播機制與梯度還原（Unbroadcasting）

當前向傳播中有廣播（broadcasting）發生時（如 3×1 向量加上 1×4 向量產生 3×4 矩陣），梯度的形狀需要「還原」到廣播前的形狀。具體來說，沿擴展維度求和即可。

本專案中的 `unbroadcast(grad, shape)` 函數實現了這個邏輯：
- 若梯度比目標形狀多 N 維，在前 N 維上求和
- 若某維度原本是 1 但現在變大，在該維度上求和

## 反向傳播的計算複雜度

設網路有 L 層，每層計算代價為 O(N)。前向傳播和反向傳播的代價大致相同，因此一次完整的訓練迭代代價約為 2L 層的計算量。反向傳播本身的時間複雜度與前向傳播同階，為 O(網路參數 × 樣本數)。

## 常見問題與挑戰

### 梯度消失（Vanishing Gradient）
網路很深時，反向傳播的梯度逐層連乘可能趨近於 0，導致前面層幾乎學不到東西。解決方法：ReLU 激活、殘差連接（ResNet）、歸一化層（Batch Normalization）、LSTM 的門控機制。

### 梯度爆炸（Exploding Gradient）
梯度連乘後變得極大，導致權重震盪或發散。解決方法：梯度裁剪（gradient clipping）、合適的權重初始化、Xavier/He 初始化。

### 病態條件數（Ill-conditioned Hessian）
梯度下降在曲率不一致的方向上收斂困難，這是為什麼需要 Adam、RMSProp 等自適應學習率優化器。

---

**上一篇**：[Gradient-Descent.md](Gradient-Descent.md)

**相關連結**：[Tensor.md](Tensor.md) | [Transformer.md](Transformer.md)