# RMSNorm（Root Mean Square Normalization）

RMSNorm 是一種輕量化的層歸一化（Layer Normalization）變體，由 Zhang & Sennrich (2019) 提出。在大型語言模型（如 LLaMA、Mistral）中廣泛使用，因為它在保持相近效果的同時，大幅降低了計算複雜度。

## 層歸一化（Layer Normalization）回顧

Layer Normalization 對神經網路中單層的所有激活值做歸一化：

$$\mu = \frac{1}{H} \sum_{i=1}^H x_i \quad \text{（均值）}$$
$$\sigma = \sqrt{\frac{1}{H} \sum_{i=1}^H (x_i - \mu)^2} \quad \text{（標準差）}$$
$$\hat{x}_i = \frac{x_i - \mu}{\sigma} \quad \text{（標準化）}$$
$$y_i = \gamma_i \hat{x}_i + \beta_i \quad \text{（仿射變換）}$$

其中 γ 和 β 是可學習的縮放和偏移參數，H 是該層的隱層維度。

LayerNorm 的優點：
- 不依賴 batch 維度，適合 batch size=1 或 RNN
- 提供良好的內在規範化，有助於訓練穩定性

缺點：
- 需要計算均值和標準差，涉及整層的減法和除法
- 對於大型模型（如 7B 參數的 LLaMA），每層都做 LN 開銷可觀

## RMSNorm 的核心思想

RMSNorm 觀察到：LayerNorm 的主要作用其實來自**均方根（Root Mean Square, RMS）**而非均值。移除均值中心化步驟，僅使用 RMS 做歸一化，效果幾乎相同但計算更快：

$$\text{RMS}(x) = \sqrt{\frac{1}{H} \sum_{i=1}^H x_i^2} \quad \text{（均方根）}$$
$$\hat{x}_i = \frac{x_i}{\text{RMS}(x)} \quad \text{（歸一化）}$$
$$y_i = \gamma_i \hat{x}_i + \beta_i$$

直覺上：均方根反映了激活值的典型幅度，除以它就將激活值規範化到一個標準尺度。均值中心化在很多情況下並非必要。

## 計算複雜度比較

LayerNorm 需要：
- 2 次累加（計算 μ 和 Σxᵢ²）
- 2 次除法（開方後除標準差）
- 1 次減法（x - μ）
- H 次乘法（逐元素除以 σ）

RMSNorm 只需要：
- 1 次累加（僅 Σxᵢ²）
- 1 次除法（直接除以 RMS）
- H 次乘法（逐元素除以 RMS）

節省了約 40% 的運算，且避免了減法運算（節省了一個 GPU 指令）。對於大型模型，這節省可觀。

## 為何 RMSNorm 有效

論文實驗表明：移除均值中心化幾乎不影響模型性能。這可能因為：
1. 現代神經網路的激活分布通常已經接近零均值
2. 殘差連接（residual connection）已經提供了 skip-gradient 路徑
3. 歸一化的核心是對尺度的規範化，均值去除並非關鍵

## 與 Group Normalization 的關係

Group Normalization（GN）將通道分組，在每組內做歸一化。當 group=1 時 GN = LayerNorm；當 group=num_channels 時 GN = InstanceNorm。GN 不依賴 batch 維度，適用於小型 batch 的視覺任務。

RMSNorm 與 GN 是互補的概念——GN 是對「對哪些維度做歸一化」的劃分，RMSNorm/LayerNorm 是對「使用什麼統計量做歸一化」的選擇。兩者可結合使用。

## 在 Transformer 中的應用

現代 Transformer 架構（如 LLaMA）在每個子層後使用 RMSNorm：
- Post-LN Transformer：LayerNorm 在殘差加法之後
- Pre-LN Transformer：LayerNorm/RMSNorm 在殘差內部（更穩定）
- RMSNorm 的位置與 Pre-LN 類似

具體來說：
```python
# 本專案 nn/optim.py 中的實現
x_normed = x / sqrt(mean(x^2) + eps)
output = gamma * x_normed
```

其中 eps=1e-5 是數值穩定性常數，防止除以零。

## 實現要點

RMSNorm 的反向傳播需要正確計算對輸入 x 的梯度。根據鏈式法則：
$$\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{1}{\text{RMS}(x)}$$

本專案的實現中計算了複雜的梯度公式以確保正確性：
```python
dx = (grad * inv_std) - (x * inv_std**3 * sum(grad * x) / N)
```

這裡 `inv_std = 1/RMS(x)`，公式源自對歸一化運算的鏈式求導。

---

**上一篇**：[Attention-Mechanism.md](Attention-Mechanism.md)

**相關連結**：[Transformer.md](Transformer.md) | [Backpropagation.md](Backpropagation.md)