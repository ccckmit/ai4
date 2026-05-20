# Attention Mechanism（注意力機制）

注意力機制是深度學習中最重要的概念之一，尤其在序列到序列（Seq2Seq）任務中。它讓模型能夠動態地「關注」輸入的不同部分，根據當前任務的需要自動學習哪些資訊是重要的。

## 從 Seq2Seq 問題說起

傳統的序列到序列模型（如機器翻譯）使用編碼器-解碼器架構：
- **編碼器（Encoder）**：將輸入序列編碼為固定維度的語境向量（context vector）
- **解碼器（Decoder）**：根據語境向量逐步生成輸出序列

瓶頸問題：語境向量是固定維度的壓縮表示，很難完整儲存長輸入的所有資訊。特別是當輸入很長時，模型往往會遺忘早期的資訊。這被稱為**資訊瓶頸**問題。

## Attention 的核心思想

注意力機制由 Bahdanau et al. (2015) 首次提出，用於解決機器翻譯中的長距離依賴問題。其核心思想是：

> 解碼時，不僅看編碼後的語境向量，還要回頭看編碼器每個時間步的輸出，根據當前解碼狀態動態決定應該「關注」哪些部分。

直覺上，翻譯時每個輸出詞可能只對應輸入的某幾個詞（而非全部）。Attention 讓模型學會這種對應關係。

## 三個關鍵組件：Q、K、V

現代注意力機制統一用 Query-Key-Value 框架描述：

- **Query（Q）**：當前要查詢的向量，「我在找什麼資訊」
- **Key（K）**：每個輸入位置的鍵向量，「我包含什麼關鍵字」
- **Value（V）**：對應的內容向量，「實際要讀取的內容」

輸出是 Value 的加权和，權重由 Query 和 Key 的相似度決定。

## Scaled Dot-Product Attention

Transformer 使用的注意力機制：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

步驟：
1. 計算 Q 和所有 K 的點積：$s_{ij} = q_i \cdot k_j$
2. 除以 $\sqrt{d_k}$ 做縮放（d_k 為 Q/K 向量維度）
3. 通過 softmax 得到注意力權重：$\alpha_{ij} = \text{softmax}(s_{ij})_j$
4. 加權求和：$o_i = \sum_j \alpha_{ij} v_j$

### 為什麼要縮放（Scaling）？

當 d_k 很大時，Q·K 的點積值會很大，使 softmax 進入饱和區域（梯度趨近於零）。除以 $\sqrt{d_k}$ 可以讓點積的方差回到合理範圍，保持 softmax 的敏感性。

## 自注意力（Self-Attention）

自注意力是輸入自己attend自己：Q、K、V 都來自同一個輸入。計算每個位置與其他所有位置的關係，讓模型捕捉輸入內部的長期依賴。

例如，在處理句子 "The cat sat on the mat" 時，"cat" 和 "sat"、"mat" 的關係可能比和 "the" 更重要。自注意力可以自動學到這些依賴。

### 與 CNN、RNN 的比較

- **CNN**：只能看到局部感受野，需要多層才能擴大
- **RNN**：理論上能看到任意距離，但實際上長期依賴難以捕捉（梯度消失/爆炸）
- **Self-Attention**：任意兩個位置直接計算相關性，O(1) 路徑長度

```mermaid
graph TD
    A[Self-Attention] --> B[任意位置直接交互]
    A --> C[路徑長度 O(1)]
    B --> D[捕捉長期依賴能力強]
    C --> D
```

## 多頭注意力（Multi-Head Attention）

單頭注意力只能捕捉一種類型的關係。多頭注意力將 Q/K/V 投影到多個子空間，分別計算注意力，再拼接：

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

每個頭可以關注不同類型的關係：
- 一個頭可能專注於語法依存
- 一個頭可能專注於語義相似
- 一個頭可能專注於指代關係

## Causal Attention（因果注意力）

在語言模型中，解碼時只能看到之前的 token，不能「偷看」未來的內容。因此需要 causal mask（也稱為 attention mask 或 future mask）：

$$\text{attention\_mask}_{ij} = \begin{cases} 0 & \text{if } j > i \text{ (future token)} \\ -\infty & \text{otherwise} \end{cases}$$

加到注意力分數上後再做 softmax，未來位置的權重變為零。

本專案 `nn/gpt.py` 中的 `CausalSelfAttention` 實現了這個機制：

```python
if T > 1:
    mask = np.triu(np.ones((T, T_k)), k=1) == 1
    attn_logits = attn_logits.masked_fill(mask, float('-inf'))
```

## Cross Attention（交叉注意力）

解碼器中另一種注意力是 cross attention：Query 來自解碼器，K/V 來源編碼器。這讓解碼器的每一步都能回頭查看編碼器的輸出。

機器翻譯時，翻譯到哪個詞就對應到原句的哪個詞，正是 cross attention 在起作用。

## 複雜度問題

標準注意力的計算和記憶體複雜度都是 O(T²)，其中 T 是序列長度。這成為長序列的瓶頸：

- 序列長度 2048：每層 4M 注意力分數
- 序列長度 32768：每層 1B 注意力分數

這催生了各種高效注意力變體：
- **Flash Attention**：IO-aware 实现，利用 GPU 記憶體層次
- **Sparse Attention**：只計算部分位置的注意力
- **Linear Attention**：核函數近似，將 O(T²) 降到 O(T)

---

**上一篇**：[Transformer.md](Transformer.md)

**相關連結**：[GPT.md](GPT.md) | [Reinforcement-Learning.md](Reinforcement-Learning.md)