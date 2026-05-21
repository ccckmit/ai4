# nn/nn.md - 神經網路層與最佳化理論

本模組 (`nn/nn.py`) 實作了神經網路的核心建構塊：網路層（Linear、Embedding）、正規化（RMSNorm）、最佳化器（Adam）、以及基礎的損失函數與激發函數。

## Module（模組基底類別）

`Module` 是所有神經網路層的抽象基底類別，採用複合模式（Composite Pattern）：

```python
class Module:
    def parameters(self):      # 收集所有可訓練參數
    def zero_grad(self):       # 清除梯度
    def __call__(self, x):     # 前向傳播
```

- `parameters()` 遞迴收集所有子模組的參數
- 支援任意深度的巢狀結構（Sequential 包含多層，每層有自己的參數）

## Linear（全連接層）

全連接層（Fully Connected Layer）是神經網路最基本的建構單元：

$$y = xW^T + b$$

- **權重矩陣 W**：形狀 `(out_features, in_features)`
- **偏置向量 b**：形狀 `(out_features,)`

初始化使用 Kaiming Uniform：
$$W \sim \mathcal{U}(-\sqrt{6 / d_{in}}, \sqrt{6 / d_{in}})$$

這種初始化方法考慮了 ReLU 的非線性特性，能有效避免梯度消失或爆炸。

## Embedding（嵌入層）

嵌入層將離散的 token ID 映射到連續向量空間：

```python
embed = Embedding(vocab_size=100, embedding_dim=16)
x = embed(tokens)  # tokens: (batch, seq_len) → (batch, seq_len, 16)
```

嵌入矩陣形狀為 `(vocab_size, embedding_dim)`，每行是一個 token 的可學習向量表示。這是 NLP 模型（如 GPT）的基礎元件。

## RMSNorm（均方根正規化）

RMSNorm（Root Mean Square Layer Normalization）是 LayerNorm 的簡化版本：

$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\text{mean}(x^2) + \epsilon}} \cdot \gamma$$

- 不使用 LayerNorm 的均值平移，只做縮放（scale）
- 計算成本更低（省去均值計算）
- LLaMA 等現代 LLM 廣泛採用

詳細理論請見 [_wiki/RMSNorm.md](../_wiki/RMSNorm.md)。

## Adam 最佳化器

Adam（Adaptive Moment Estimation）結合了 Momentum 和 RMSProp 的優點：

1. **一階動量（momentum）**：$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
2. **二階動量（RMSProp）**：$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$
3. **偏差校正**：$\hat{m}_t = m_t / (1-\beta_1^t)$，$\hat{v}_t = v_t / (1-\beta_2^t)$
4. **參數更新**：$\theta_t = \theta_{t-1} - \eta \cdot \hat{m}_t / (\sqrt{\hat{v}_t} + \epsilon)$

預設超參數 $\beta_1=0.9$、$\beta_2=0.999$、$\epsilon=10^{-8}$，與原始論文一致。

詳細理論請見 [_wiki/Adam-Optimizer.md](../_wiki/Adam-Optimizer.md)。

## 激發函數

- **ReLU（Rectified Linear Unit）**：$f(x) = \max(0, x)$
  - 梯度為 0（x<0）或 1（x>0），緩解梯度消失
  - 廣泛用於隱藏層
- **Tanh（雙曲正切）**：$f(x) = \tanh(x) \in (-1, 1)$
  - 零中心，適合 RNN

詳細理論請見 [_wiki/Activation-Function.md](../_wiki/Activation-Function.md)。

## 損失函數

**MSE Loss（均方誤差）**：$\mathcal{L} = \frac{1}{n} \sum (y_{pred} - y_{true})^2$

用於回歸任務。梯度與預測誤差成正比。

詳細理論請見 [_wiki/Loss-Function.md](../_wiki/Loss-Function.md)。

## Sequential（序列容器）

Sequential 按順序串聯多個層：

```python
model = Sequential(
    Linear(784, 256),  # 輸入 784 → 隱藏層 256
    ReLU(),
    Linear(256, 10),   # 隱藏層 256 → 輸出 10
)
```

前向傳播時依序呼叫每個子模組。參數收集由 Module 的遞迴機制自動完成。

---

**相關連結**：[tensor.md](tensor.md) | [gpt.md](gpt.md) | [Backpropagation.md](../_wiki/Backpropagation.md) | [Gradient-Descent.md](../_wiki/Gradient-Descent.md)
