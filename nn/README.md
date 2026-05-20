# nn/ — DIY 神經網路框架

一個純 NumPy 實現的神經網路框架，支援自動微分、梯度下降和 Transformer 架構。用於學習深度學習內部原理。

## 核心模組

| 檔案 | 說明 |
|------|------|
| `tensor.py` | 自動微分張量（含反向傳播） |
| `optim.py` | Module, Linear, Embedding, RMSNorm, Adam |
| `gpt.py` | GPT 模型（多層 Transformer + KV Cache） |
| `chargpt.py` | 語言模型訓練迴圈 |
| `chargpt_demo.py` | 訓練 CharGPT 生成中文名字 |

## Tensor 自動微分

```python
from nn import Tensor, cat

a = Tensor([[1, 2], [3, 4]], requires_grad=True)
b = Tensor([[5, 6], [7, 8]], requires_grad=True)
c = a @ b                    # 矩陣乘法
loss = c.sum()
loss.backward()             # 反向傳播
print(a.grad)                # 梯度已計算
```

支援：`+`, `*`, `@`, `relu`, `softmax`, `cross_entropy`, `transpose`, `reshape`, `sum` 等。

## 模型建構

```python
from nn import GPT

model = GPT(vocab_size=100, block_size=32, n_layer=2, n_embd=32, n_head=4)
logits, caches = model(token_ids, kv_caches=None)
```

## 訓練

```bash
uv run python -m nn.chargpt_demo
```

## 理論背景

- [\_wiki/Tensor.md](tensor.md) — 自動微分原理
- [\_wiki/Backpropagation.md](../_wiki/Backpropagation.md) — 反向傳播演算法
- [\_wiki/Transformer.md](../_wiki/Transformer.md) — Transformer 架構
- [\_wiki/GPT.md](../_wiki/GPT.md) — GPT 語言模型
- [\_wiki/RMSNorm.md](../_wiki/RMSNorm.md) — RMSNorm 歸一化
- [\_wiki/Gradient-Descent.md](../_wiki/Gradient-Descent.md) — 梯度下降與 Adam

## 模組結構

```
nn/
  tensor.py    # Tensor 類 + 反向傳播
  optim.py     # 網路層（Linear, Embedding, RMSNorm）+ Adam
  gpt.py       # GPT, Block, CausalSelfAttention, MLP
  chargpt.py   # 訓練循環（train_model, generate_samples）
 chargpt_demo.py
```