# Transformer

Transformer 是 2017 年《Attention Is All You Need》論文中提出的模型架構，完全基於注意力機制（attention mechanism），完全拋棄了傳統的循環神經網路（RNN）和卷積神經網路（CNN）。它的核心創新是**自注意力機制（Self-Attention）**和**位置編碼（Positional Encoding）**，使得並行計算成為可能，並在幾乎所有自然語言處理（NLP）任務上達到 state-of-the-art。

## 從 RNN 到 Transformer

傳統 RNN（如 LSTM、GRU）處理序列資料時存在两个根本問題：

1. **順序依賴**：必須等前面計算完成才能處理後續內容，無法平行化
2. **長期依賴**：梯度在時間維度上反向傳播時，長時間之前的信號會被稀釋或爆炸

Transformer 完全拋棄了序列遞進結構，改用注意力機制直接建立任意兩個位置之間的依賴關係，實現了真正的平行計算，並能更有效地捕捉長距離依賴。

## 注意力機制的數學原理

 Transformer 使用的是 **Scaled Dot-Product Attention**：

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

其中：
- **Q（Query）**：查詢向量，代表「我在找什麼」
- **K（Key）**：鍵向量，代表「我包含什麼資訊」
- **V（Value）**：數值向量，代表「實際的內容」

步驟：
1. 計算 Q 和 K 的點積，得到相關性分數
2. 除以 $\sqrt{d_k}$ 做縮放（防止梯度消失）
3. 通過 softmax 得到注意力權重（機率分布）
4. 加權求和 V，得到輸出

直覺上：每個輸出位置是所有輸入價值的加权和，權重由 Query 和 Key 的相似度決定。

## 多頭注意力（Multi-Head Attention）

只做一次注意力稱為單頭。多頭注意力是將 Q、K、V 分別透過線性投影到 h 個不同的子空間（每個頭有 $d_k = d_{model}/h$ 維），分別計算注意力後 concatenate，再經過一個線性投影：

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

多頭注意力的優點：
- 每個頭可以關注不同類型的關係（如句法、語義、語境）
- 提供更豐富的表示能力
- 增加模型穩健性（壞了一個頭其他仍可運作）

## 位置編碼（Positional Encoding）

Transformer 本身沒有循環結構，無法區分序列中不同位置的元素。為了解決這個問題，論文中引入了位置編碼，將位置信息添加到輸入嵌入中：

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

選擇正弦/餘弦函數的好處是：任意兩個位置的相對距離可以用線性組合表示，且可以推廣到訓練時未見過的序列長度。

後續工作也有使用可學習的位置編碼（ Learned Positional Encoding）或旋轉位置編碼（RoPE）等更先進的方法。

## Transformer 整體架構

標準的 Transformer 由編碼器（Encoder）和解碼器（Decoder）堆疊組成：

### 編碼器（Encoder）
每層包含兩個子層：
1. **Multi-Head Self-Attention**：對輸入序列做自注意力
2. **Feed-Forward Network（FFN）**：簡單的兩層全連接網路

每個子層周圍有**殘差連接（Residual Connection）**和**層歸一化（Layer Normalization）**：

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

殘差連接緩解梯度消失，層歸一化穩定訓練。

### 解碼器（Decoder）
與編碼器類似的結構，但多了一個子層：
1. **Masked Multi-Head Self-Attention**：防止看到未來的位置（因果遮蔽）
2. **Cross-Attention**：Query 來自解碼器，K/V 來自編碼器輸出
3. **Feed-Forward Network**

### 堆疊層數
原始論文使用 N=6 層編碼器和 N=6 層解碼器。現代大型模型（如 GPT-3）使用 96 層或更多。

## 訓練與推理的差異

### 訓練（Training）
- 輸入是完整的目標序列（如翻譯任務中的原文和譯文）
- 解碼器使用**teacher forcing**：輸入是正確的上一個 token
- 訓練高效，可以平行處理整個序列

### 推理（Inference）
- 只能逐步生成（autoregressive），每次預測下一個 token
- 當前輸出要作為下一個輸入
- 計算量是 O(T²)，T 為序列長度

## KV Cache 加速推理

推理時每個新 token 只需要 attend 到之前所有 token，但完整重新計算代價高昂。**KV Cache** 的思想是：
- 儲存並復用之前 token 的 Key 和 Value 向量
- 每個新 token 只需計算自己的 Q，與快取中的 K/V 組合
- 將新計算的 K/V 也加入快取

代價：記憶體消耗隨序列長度線性增長（但時間複雜度降為 O(1) 每新增 token）。

本專案 `nn/gpt.py` 中的 `CausalSelfAttention` 實現了 KV Cache：當傳入 `kv_cache` 參數時，會先拼接歷史 K/V 與當前 K/V，再計算注意力。GPT 類的 `__call__` 方法也會傳回更新後的 `new_caches`，供下一個 forward pass 使用。

## 複雜度與改進

### 標準注意力的瓶頸
標準 self-attention 的時間和空間複雜度都是 O(T²·d)，其中 T 是序列長度，d 是模型維度。這在長序列上成為瓶頸。

### 改進方案
- **Flash Attention**：IO-aware 的精確注意力實現，利用 GPU 記憶體層級加速
- **Sparse Attention**：只計算部分位置的注意力（如 BigBird）
- **Linear Attention**：將 O(T²) 降到 O(T)（如 Performer、Linear Transformer）
- **Ring Attention**：分散式長序列計算

## 演化脈絡

```
Transformer (2017) 
  ├── GPT 系列（OpenAI）：只用解碼器，強調語言建模
  │     ├── GPT-2 (2019)：15億參數
  │     ├── GPT-3 (2020)：1750億參數，In-Context Learning
  │     └── GPT-4 (2023)：多模態
  ├── BERT (2018)：只用編碼器，強調理解任務
  ├── T5 (2019)：編碼器-解碼器統一框架
  └── LLaMA (2023)：開源 LLM 里程碑
```

本專案的 `nn/gpt.py` 實現了一個簡化的 GPT 模型，包含 Multi-Head Self-Attention、MLP、Transformer Block 和 KV Cache 支援，可以作為學習 Transformer 架構的良好起點。

---

**上一篇**：[Attention-Mechanism.md](Attention-Mechanism.md)

**相關連結**：[GPT.md](GPT.md) | [Backpropagation.md](Backpropagation.md)