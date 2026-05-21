# nn/chargpt.md - 字元級 GPT 訓練與生成

本模組實作了**字元級語言模型**（Character-Level Language Model）的訓練和生成流程。以 GPT 架構為基礎，在字元序列上進行自回歸建模。

## 從字元到序列

字元級語言模型將文字視為字元序列。給定一個字串 "hello"，對應的 token IDs 為：

```python
vocab = ['h', 'e', 'l', 'o']  # 詞彙表（vocab）
chars = ['h', 'e', 'l', 'l', 'o']
ids   = [0, 1, 2, 2, 3]       # token IDs
```

### 字元級 vs 子詞級

| 特性 | 字元級（本專案） | 子詞級（BPE/WordPiece） |
|------|----------------|----------------------|
| 詞彙表大小 | ~100（極小） | 32K-100K |
| 序列長度 | 長（字元數） | 短（token 數） |
| 語義單位 | 字母 | 詞根/詞綴 |
| 適合場景 | 教學/簡單任務 | 生產級 LLM |

字元級模型的優點是詞彙表小、實作簡單，適合教學和原型開發。

## 訓練流程

### 資料準備

從文字檔讀取所有字元，建立詞彙表：

```python
text = open("input.txt").read()
chars = sorted(list(set(text)))
vocab_size = len(chars)  # 通常 ~100
```

### 批次資料

將長文字序列切分成多個區塊（chunk），每個區塊長度為 `block_size`（如 256）：

```
text = "To be or not to be, that is the question..."
chunk1 = "To be or not to be, that is the q"
chunk2 = "hat is the question..."
```

每個樣本是一個 `(block_size,)` 的整數陣列。目標（target）是向右偏移一個位置的相同陣列：

```python
x = text[0:block_size]   # "To be or not to b"
y = text[1:block_size+1] # "o be or not to be"
```

這是語言模型的自回歸訓練方式：每個位置預測下一個字元。

### 訓練迴圈

```python
for step in range(max_steps):
    x, y = sample_batch(data)
    logits = model(x)
    loss = cross_entropy(logits, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## 文字生成（取樣）

訓練完成後，從 prompt 開始逐步生成：

```python
def generate(model, prompt, steps):
    for _ in range(steps):
        logits = model(prompt)
        next_token = sample(logits[-1])  # 取最後位置
        prompt = cat([prompt, next_token])
    return prompt
```

### 取樣策略

1. **貪婪取樣（Greedy）**：取機率最高的 token
2. **隨機取樣**：根據機率分布隨機選取，引入多樣性
3. **Temperature 控制**：softmax 前縮放 logits：

$$P_i = \frac{\exp(logit_i / T)}{\sum_j \exp(logit_j / T)}$$

- $T \to 0$：趨近貪婪（確定性）
- $T = 1$：原始分布
- $T > 1$：分布更均勻（更多隨機性）

## 損失曲線與過擬合

字元級 GPT 訓練時，loss 通常從 ~4.0（均勻分布的交叉熵）下降到接近 0。若 loss 趨近於 0，表示模型已經「記住」了訓練資料（過擬合），但對未見過的文字仍然可以產生合理的字元序列。

詳細理論請見 [_wiki/GPT.md](../_wiki/GPT.md)。

---

**相關連結**：[gpt.md](gpt.md) | [nn.md](nn.md) | [tensor.md](tensor.md)
