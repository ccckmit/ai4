# Activation Function（激活函數）

激活函數（activation function）是神經網路中不可或缺的非線性變換元件。若沒有激活函數，多層線性網路的組合仍然等價於一個單層線性變換，失去了深層表達的意義。激活函數的選擇直接影響網路的收斂速度、表達能力和訓練穩定性。

## 為什麼需要非線性

考慮一個 $L$ 層全連接網路，每層都是線性變換 $h^{(l)} = W^{(l)} h^{(l-1)} + b^{(l)}$。將所有層合併：

$$h^{(L)} = W^{(L)} (\cdots (W^{(2)} (W^{(1)} x + b^{(1)}) + b^{(2)}) \cdots ) + b^{(L)}$$

展開後這等價於一個單層線性變換：

$$h^{(L)} = (W^{(L)} \cdots W^{(2)} W^{(1)}) x + (\text{偏差組合})$$

因此，不含非線性的深層網路並未比單層網路有更強的表達能力。非線性激活函數打破了這種線性疊加的局限性，使網路能夠逼近任意複雜的函數（通用近似定理，Universal Approximation Theorem）。

**通用近似定理**：一個包含至少一個隱藏層的前饋神經網路，搭配適當的非線性激活函數，可以在任意精確度下逼近任何定義在 $\mathbb{R}^n$ 緊緻子集上的連續函數。

## Sigmoid 函數

Sigmoid 函數將輸入映射到 $(0, 1)$ 區間：

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

$$\sigma'(x) = \sigma(x) \cdot (1 - \sigma(x))$$

Sigmoid 是早期神經網路中最常用的激活函數，現在已較少使用。其問題：

1. **飽和梯度（Saturated Gradient）**：當 $|x|$ 很大時，$\sigma'(x) \approx 0$，梯度幾乎為零，造成學習停滯
2. **非零中心（Not Zero-Centered）**：輸出恆為正（0 到 1），導致後續層的梯度更新方向單一，收斂效率低
3. **指數計算**：計算代價比 ReLU 高

## Tanh 函數

Tanh（雙曲正切）是 Sigmoid 的縮放版本，將輸入映射到 $(-1, 1)$：

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1$$

$$\tanh'(x) = 1 - \tanh^2(x)$$

Tanh 是零中心的（zero-centered），這解決了 Sigmoid 的一個主要缺點。然而 Tanh 仍存在飽和問題：當 $|x|$ 較大時，梯度趨近於零。本專案 `nn/nn.py:79-83` 的 `Tanh` 類包裝了 `tensor.py` 中的 `.tanh()` 方法，其反向傳播實作為：

```python
self.grad += out.grad * (1 - out.data ** 2)
```

## ReLU 函數

ReLU（Rectified Linear Unit，修正線性單元）是現代深度學習的默認激活函數：

$$\text{ReLU}(x) = \max(0, x)$$

$$\text{ReLU}'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{otherwise} \end{cases}$$

### 優點

1. **非飽和性**：在 $x > 0$ 區域梯度恆為 1，不會飽和，大幅緩解梯度消失問題
2. **計算簡單**：只需一個 max 操作，無需指數計算
3. **稀疏激活**：約一半的神經元輸出為零，產生自然的稀疏表達（sparse representation）
4. **經驗收斂快**：在實踐中，ReLU 網路的收斂速度遠快於 Sigmoid/Tanh

### 缺點

1. **神經元死亡（Dying ReLU）**：若某神經元的輸入始終為負，其梯度始終為零，權重停止更新，神經元永久死亡
2. **非零中心**：輸出恆非負（所有正值，零），可能影響後續層的訓練效率
3. **無上界**：輸出可以非常大，可能導致激活值爆炸

本專案 `nn/nn.py:72-76` 的 `ReLU` 類包裝了 `tensor.py` 中的 `.relu()` 方法，其梯度為簡單的門控：

```python
self.grad += out.grad * (self.data > 0)
```

## Leaky ReLU

為了解決 Dying ReLU 問題，Leaky ReLU 在負半軸引入一個小斜率：

$$\text{LeakyReLU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha x & \text{otherwise} \end{cases}$$

其中 $\alpha$ 通常取 0.01。負半軸的微小梯度確保神經元即使長時間處於負值也能緩慢恢復。

後續變體還包括 **Parametric ReLU (PReLU)**，將 $\alpha$ 改為可學習的參數；以及 **Randomized Leaky ReLU (RReLU)**，訓練時從均勻分布中隨機採樣 $\alpha$。

## GELU 函數

GELU（Gaussian Error Linear Unit）是近年來在 Transformer 架構中廣泛使用的激活函數，由 Hendrycks & Gimpel（2016）提出：

$$\text{GELU}(x) = x \cdot \Phi(x)$$

其中 $\Phi(x)$ 是標準高斯分布的累積分布函數（CDF）。實用中常用近似公式：

$$\text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)$$

GELU 是 ReLU 的平滑版本。相比 ReLU，GELU 在 x=0 處連續可導（ReLU 在 0 處不可導），且負值區域保留微小梯度（不像 ReLU 完全截斷）。

本專案 `nn/gpt.py:88-89` 的 MLP 層使用的是 `relu()`，但 GPT 原論文中使用的是 GELU。GELU 的隨機理解是——它根據輸入值的大小以機率 $\Phi(x)$ 進行捨棄：輸入越大，保留機率越高。

## Softmax 函數

Softmax 是分類任務輸出層的標準激活函數，將 K 個實數向量轉換為機率分布：

$$\text{softmax}(x)_i = \frac{e^{x_i}}{\sum_{j=1}^K e^{x_j}}$$

特點：
- 輸出所有元素和為 1，且 $0 < \text{softmax}(x)_i < 1$
- 具有平移不變性：$\text{softmax}(x + c) = \text{softmax}(x)$（加上常數後分子分母抵消）
- 對數 softmax：$\log\text{softmax}(x)_i = x_i - \log\sum\exp(x_j)$，常用於數值穩定性

Softmax 的雅可比（Jacobian）矩陣：

$$\frac{\partial \text{softmax}(x)_i}{\partial x_j} = \text{softmax}(x)_i (\delta_{ij} - \text{softmax}(x)_j)$$

其中 $\delta_{ij}$ 是 Kronecker delta。本專案 `nn/tensor.py:200-216` 的 `.softmax()` 方法實現了高效的向量化梯度計算：

```python
self.grad += s * (grad_s - np.sum(grad_s * s, axis=axis, keepdims=True))
```

而非直接計算完整的雅可比矩陣（後者需要 $O(K^2)$ 的記憶體和計算量）。

## 梯度消失問題（Vanishing Gradient Problem）

深度神經網路訓練的核心難題之一。反向傳播時，梯度從輸出層向輸入層逐層連乘：

$$\frac{\partial L}{\partial W^{(1)}} = \frac{\partial L}{\partial h^{(L)}} \cdot \frac{\partial h^{(L)}}{\partial h^{(L-1)}} \cdots \frac{\partial h^{(2)}}{\partial h^{(1)}} \cdot \frac{\partial h^{(1)}}{\partial W^{(1)}}$$

當激活函數的導數 $\leq 1$（如 Sigmoid 最大導數為 0.25，Tanh 最大導數為 1），且權重初始化較小時，梯度指數級衰減，深層無法學習。

### 緩解策略

1. **ReLU 系激活函數**：正半軸梯度為 1，不壓縮梯度
2. **殘差連接（Residual Connection）**：$x_{l+1} = x_l + F(x_l)$，梯度可經由 identity path 直接回傳
3. **歸一化層**：Batch Normalization / Layer Normalization 穩定激活值分布
4. **合適的初始化**：Xavier（Glorot）初始化、He 初始化，保證各層輸出方差穩定
5. **門控機制**：LSTM 等使用門控（gating）控制梯度流

## 飽和 vs 非飽和激活函數

### 飽和（Saturating）

當輸入趨向 $\pm\infty$ 時，導數趨於零的激活函數。

| 函數 | 飽和區域 | 問題 |
|------|---------|------|
| Sigmoid | $x \to \infty$ 或 $x \to -\infty$ | 雙向飽和，非零中心 |
| Tanh | $x \to \infty$ 或 $x \to -\infty$ | 雙向飽和，零中心 |

### 非飽和（Non-Saturating）

輸入趨向 $\pm\infty$ 時導數不趨於零。

| 函數 | 特性 | 問題 |
|------|------|------|
| ReLU | 正半軸導數 1 | Dying ReLU |
| Leaky ReLU | 負半軸導數 $\alpha$ | 超參數選擇 |
| GELU | 平滑且處處可導 | 計算稍複雜 |

## 激活函數的選擇準則

| 場景 | 建議 |
|------|------|
| CNN 隱藏層 | ReLU 或 Leaky ReLU |
| RNN/Transformer | GELU 或 Swish/SiLU |
| 輸出層：回歸 | 無激活函數（線性輸出） |
| 輸出層：二分類 | Sigmoid |
| 輸出層：多分類 | Softmax |
| 輸出層：多標籤 | 逐元素 Sigmoid |
| 自編碼器隱藏層 | Tanh 若輸入歸一化到 [-1, 1] |

## 其他重要激活函數

### Swish / SiLU

$$\text{Swish}(x) = x \cdot \sigma(\beta x)$$

其中 $\beta$ 可學習或固定（$\beta=1$ 時稱為 SiLU）。Swish 具有下界無上限、平滑、非單調等特性，在許多任務上優於 ReLU。

### ELU（Exponential Linear Unit）

$$\text{ELU}(x) = \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}$$

負值區域呈指數衰減而非截斷，使輸出均值接近零。

### Mish

$$\text{Mish}(x) = x \cdot \tanh(\ln(1 + e^x))$$

Mish 是 Swish 的變體，被提出用於電腦視覺任務，但計算成本較高。

## 激活函數的演化

```
感知機（階躍函數）
  └── 多層感知機（Sigmoid / Tanh） 
       └── 深度學習（ReLU, 2011-2013）
            ├── Leaky ReLU (2013)
            ├── PReLU (2015)
            ├── ELU (2015)
            ├── Swish / SiLU (2017)
            └── GELU (2016, 流行於 Transformer 時代, 2017-)
```

ReLU 的出現是深度學習復興的關鍵轉折點，而 GELU 在 Transformer 時代扮演了類似的角色。目前 ReLU 仍然是 CNN 架構的預設選擇，GELU/SiLU 則主導了 LLM 架構。

## 激活函數的梯度流分析

激活函數對梯度流的影響可透過「梯度傳播因子」量化。對於前向 $a=f(z)$，後向 $\delta = f'(z) \cdot \delta_{\text{up}}$，梯度傳播因子為 $f'(z)$。

```mermaid
graph LR
    subgraph 前向
        Z[z = Wx + b] --> A[a = f(z)] --> L[Loss]
    end
    subgraph 反向
        L --> dA[dL/da] --> dZ[dL/dz = f'(z) * dL/da] --> dW[dL/dW]
    end
```

設網路有 L 層，輸入層梯度約為：

$$\left\|\frac{\partial L}{\partial z^{(1)}}\right\| \approx \left\| \frac{\partial L}{\partial z^{(L)}} \right\| \cdot \prod_{l=2}^L \| f'(z^{(l)}) \|$$

**Sigmoid**：$f'(z)$ 最大值僅 0.25，且大部分區域遠小於 0.25。L=10 層後梯度乘積 $\leq 0.25^{10} \approx 10^{-6}$，完全消失。

**Tanh**：$f'(z)$ 最大值為 1，但飽和區域導數接近 0。有「窗口期」——只有在 $z \in [-2, 2]$ 範圍內梯度才有效。

**ReLU**：$f'(z) \in \{0, 1\}$。成功路徑的梯度完全不衰減。統計上，每層約半數神經元死亡（輸出 0），但存活的神經元保持完整梯度。

**GELU**：$f'(z)$ 在 0 處約為 0.5，負半軸漸進於 0，正半軸漸進於 1。性質介於 ReLU 和 Sigmoid 之間。

## Swish 的無界性與正則化

Swish/SiLU 的一個有趣特性是在 $(-\infty, 0)$ 區間為負值且最小值約為 $-0.278$。這引入了隱式的 L2 正則化效果：小幅負值的梯度在反向傳播中起到阻尼作用。

## 激活函數的計算成本比較

| 函數 | 加法 | 乘法 | 指數 | 條件分支 | 相對延遲 |
|------|------|------|------|---------|---------|
| ReLU | 0 | 0 | 0 | 1 | 1× |
| Leaky ReLU | 1 | 1 | 0 | 1 | ~1× |
| Sigmoid | 2 | 1 | 1 | 0 | ~4× |
| Tanh | 2 | 3 | 2 | 0 | ~5× |
| GELU (近似) | 4 | 5 | 0 | 1 (tanh) | ~3× |
| Swish/SiLU | 2 | 1 | 1 | 0 | ~4× |

在硬體（GPU/TPU）上，ReLU 的優勢更明顯——條件分支可高效向量化，無需浮點運算。但現代硬體的矩陣乘法（Linear/Conv 層）佔總計算量的 99% 以上，激活函數的計算成本通常可忽略。

## 非單調激活函數

大多數激活函數是單調的（>0 時單調增）。Swish 的獨特之處在於其在 x < 0 時非單調——先降至負值再上升至零。

非單調性的好處：當某些神經元的輸入為負時，Swish 產生較大負值（遠離飽和區），梯度較大，幫助參數更新將該神經元的操作點推向非飽和區。相比之下，ReLU 直接將負值截斷為 0，剝奪了這些神經元的學習機會。

## 激活函數與權重初始化的交互

權重初始化需要考慮激活函數的性質，以維持前向激活值和反向梯度的方差。

### Xavier/Glorot 初始化

適用於 Sigmoid 和 Tanh，要求各層輸入和輸出的方差一致：

$$W \sim \mathcal{U}\left(-\sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}, \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}\right)$$

對於 Tanh，其 $f'(0)=1$，Xavier 初始化能維持 $\text{Var}[z^{(l)}] = \text{Var}[z^{(l-1)}]$。

### He/Kaiming 初始化

適用於 ReLU 系列（ReLU、Leaky ReLU、PReLU）：

$$W \sim \mathcal{N}\left(0, \sqrt{\frac{2}{n_{\text{in}}}}\right)$$

ReLU 將一半的神經元輸出設為零，導致方差減半。He 初始化透過將標準差放大 $\sqrt{2}$ 倍來補償這個影響。

| 激活函數 | 推薦初始化 | 方差保持條件 |
|---------|-----------|------------|
| Sigmoid | Xavier | $f'(0) = 0.25$，實際會衰減 |
| Tanh | Xavier | $f'(0) = 1$ |
| ReLU | He | $\mathbb{E}[a^2] = \frac{1}{2} \mathbb{E}[z^2]$ |
| Leaky ReLU | He 修改 | 取決於 $\alpha$ |
| GELU | He 或更小的 std | 0 處梯度約 0.5 |
| Swish | He | 類似 GELU |

## 自歸一化激活函數：SELU

SELU（Scaled Exponential Linear Unit, Klambauer et al., 2017）設計目標是「自歸一化」——即使不使用 Batch Normalization，深層網路的激活值也能自動保持零均值、單位方差：

$$\text{SELU}(x) = \lambda \begin{cases} x & \text{if } x > 0 \\ \alpha(e^x - 1) & \text{otherwise} \end{cases}$$

其中 $\lambda \approx 1.0507, \alpha \approx 1.6733$。

SELU 的理論基礎是透過選擇特定的 $(\lambda, \alpha)$ 使激活值分布對均值和方差具有吸引定點（attractive fixed point）。然而 SELU 對權重初始化和 dropout 類型較敏感，實務中未廣泛取代 ReLU。

---

**上一篇**：[Convolutional-Neural-Network.md](Convolutional-Neural-Network.md)

**相關連結**：[Backpropagation.md](Backpropagation.md) | [Transformer.md](Transformer.md) | [Loss-Function.md](Loss-Function.md)
