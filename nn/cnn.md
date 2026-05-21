# nn/cnn.md - 卷積神經網路理論

本模組實現了卷積神經網路（Convolutional Neural Network, CNN）的核心層：Conv2d、MaxPool2d、AvgPool2d、Flatten、BatchNorm2d、Dropout2d。CNN 是電腦視覺領域的基石架構。

## 卷積運算（Convolution）

卷積層的核心是**滑動視窗**：一個小型的 kernel（過濾器）在輸入上滑動，每個位置計算點積：

$$y[i, j] = \sum_{m=0}^{k_h-1} \sum_{n=0}^{k_w-1} x[i+m, j+n] \cdot w[m, n] + b$$

### 超參數

- **kernel_size**：過濾器大小（如 3×3、5×5）
- **stride**：滑動步長（=1 則輸出與輸入同解析度）
- **padding**：邊緣補零，控制輸出尺寸
- **channels**：輸入/輸出通道數

### 輸出尺寸計算

$$H_{out} = \left\lfloor \frac{H_{in} + 2 \times padding - kernel\_size}{stride} \right\rfloor + 1$$

卷積的關鍵特性：
- **局部連接**：每個輸出只與局部輸入相連
- **參數共享**：同一個 kernel 掃過整個輸入
- **平移不變性**：特徵偵測與位置無關

## MaxPool2d / AvgPool2d（池化層）

池化層對局部區域進行降採樣：

- **MaxPool**：取區域內最大值
  - 保留最強烈的特徵響應
  - 對微小位移具有不變性
- **AvgPool**：取區域內平均值
  - 保留整體資訊
  - 常用於全域平均池化（替代全連接層）

池化層沒有可學習參數，主要功能是降低空間維度、減少計算量、控制過擬合。

## BatchNorm2d（批次正規化）

批次正規化讓層輸入的分布保持穩定：

$$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y = \gamma \hat{x} + \beta$$

- $\mu_B, \sigma_B$：當前 batch 的均值和標準差
- $\gamma, \beta$：可學習的縮放和偏移參數
- 訓練時使用 batch 統計量；推理時使用累積的 running mean/var

BatchNorm 的效果：加速收斂、允許更高學習率、有輕微正則化效果。

## Dropout2d（隨機丟棄）

Dropout 是有效的正則化技術，訓練時隨機將部分神經元輸出設為零：

- Dropout：隨機丟棄個別神經元（用於全連接層）
- Dropout2d：隨機丟棄整個通道（用於卷積層）

後者在 CNN 中更合理，因為相鄰空間位置高度相關，單獨丟棄個別像素效果有限。

推理時 Dropout 關閉，所有神經元都參與計算（權重乘以保留機率 $\frac{1}{1-p}$）。

## Flatten（展平層）

Flatten 將多維特徵圖展平為一維向量，連接 CNN 和後續的全連接層：

```python
# 輸入: (batch, channels, height, width)
# 輸出: (batch, channels * height * width)
```

## CNN 典型架構

```mermaid
graph LR
    Input[(輸入影像)] --> Conv[Conv2d + ReLU]
    Conv --> Pool[MaxPool2d]
    Pool --> Conv2[Conv2d + ReLU]
    Conv2 --> Pool2[MaxPool2d]
    Pool2 --> Flat[Flatten]
    Flat --> FC[Linear + ReLU]
    FC --> Out[Linear + Softmax]
```

這種「卷積 → 池化 → 卷積 → 池化 → 全連接」的模式是 LeNet/AlexNet 時代以來的經典設計。

詳細理論請見 [_wiki/Convolutional-Neural-Network.md](../_wiki/Convolutional-Neural-Network.md)。

---

**相關連結**：[nn.md](nn.md) | [tensor.md](tensor.md)
