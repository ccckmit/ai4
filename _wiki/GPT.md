# GPT（Generative Pre-trained Transformer）

GPT 是 OpenAI 开发的大型語言模型系列，全稱為「生成式預訓練變換器」。從 GPT-1（2018）到 GPT-4（2023），GPT 系列推動了大型語言模型的發展，奠定了當今生成式 AI 的基礎。

## 歷史脈絡

- **GPT-1 (2018)**：首次提出「預訓練+微調」範式，在 BooksCorpus 上訓練，證明了 Transformer 解碼器的強大語言建模能力
- **GPT-2 (2019)**：擴大到 15 億參數，展示了驚人的零樣本（zero-shot）能力
- **GPT-3 (2020)**：擴大到 1750 億參數，引入 In-Context Learning
- **GPT-4 (2023)**：多模態能力，支持圖像輸入，推理能力顯著提升

## 架構：僅解碼器的 Transformer

GPT 只使用 Transformer 的**解碼器**部分，拋棄了編碼器。這是因為 GPT 是自回歸語言模型——根據前面的 token 預測下一個 token，只需要單向（causal）注意力。

與 BERT（編碼器-only）的比較：
- **BERT**：雙向注意力，適合理解任務（如分類、問答）
- **GPT**：單向（因果）注意力，適合生成任務

GPT 的核心是「next token prediction」——給定前面的文字序列，預測下一個最可能的 token。

## 預訓練目標

GPT 的預訓練目標是**因果語言建模（Causal Language Modeling, CLM）**，也稱為**自回歸語言建模**：

$$\mathcal{L} = -\sum_{t=1}^T \log P(x_t \mid x_{<t}; \theta)$$

最大化每個位置在給定前文時預測正確 token 的對數機率。整個語料庫的總 loss 是所有 token 的負對數似然之和。

這與 BERT 的掩碼語言建模（Masked Language Modeling）形成對比——BERT 可以看到上下文（雙向），而 GPT 只能看到過去（單向）。

## 語言建模的直覺

語言模型本質上是在學習一個條件機率分佈 $P(x_t | x_1, x_2, ..., x_{t-1})$。訓練時：
1. 輸入一個 token 序列
2. 模型預測下一個 token
3. 計算預測與真實的交叉熵 loss
4. 反向傳播更新參數

訓練信號是「下一個 token 應該是什麼」。這看起來很簡單，但當模型規模夠大、訓練資料夠多時，語言模型會自發地湧現出各种能力——問答、翻譯、程式生成、推理等。

## In-Context Learning（上下文內學習）

GPT-3 的關鍵發現：大型語言模型可以在不進行梯度更新的情況下，根據輸入中的少數範例快速適應新任務。這稱為 In-Context Learning 或 Few-Shot Learning。

原理：用戶在 prompt 中提供任務描述和幾個範例，模型根據這些「上下文」理解任務並生成回答：

```
翻譯成法文：
輸入: Hello
輸出: Bonjour

輸入: Good morning
輸出:
```

模型會根據前面的範例「 Bonjour」理解翻譯任務，生成「Bonjour」的翻譯（但這裡是英文→法文）。

## 溫度與 Top-p 採樣

語言模型的輸出是機率分佈。生成文字時需要從這個分佈中採樣：

- **Temperature**：控制隨機性。T=0 時總選最高機率的 token（貪心）；T=1 時按原始分佈；T>1 時增加隨機性
- **Top-k**：限制每次只在機率最高的 k 個 token 中採樣
- **Top-p（Nucleus Sampling）**：選擇累積機率剛好超過 p 的最小集合中採樣

溫度太低：輸出確定性高、可能重複。溫度太高：輸出混亂、偏離主題。實務上常用 T=0.7~1.0 配合 top_p=0.9。

## KV Cache 加速推理

自回歸生成的問題：每生成一個新 token，都需要重新計算所有之前 token 的注意力（否則無法 attend 到它們）。這是 O(T²) 的時間複雜度。

KV Cache 的解法：
1. 第一次 forward：計算並快取所有 K、V
2. 之後每步：只計算新 token 的 Q，利用快取計算新 token 與所有歷史的注意力
3. 將新的 K、V 加入快取

代價：記憶體 O(T)，時間從 O(T²) 降到 O(T)（攤銷後）。

本專案 `nn/gpt.py` 實現了 KV Cache：`CausalSelfAttention` 接收可選的 `kv_cache`，拼接歷史 K/V 與當前 K/V 後計算注意力。GPT 的 `__call__` 返回 `new_caches` 供下次調用使用。

## Transformer 時代的 LLM 訓練

大型語言模型的訓練分三階段：
1. **預訓練（Pretraining）**：在海量文本上做因果語言建模
2. **指令微調（Instruction Tuning）**：在高質量問答對上微調
3. **人類反饋強化學習（RLHF）**：用人類偏好数据训练 reward model，再做 PPO 優化

本專案的 `chargpt_demo.py` 只實現了預訓練階段——在 names.txt 資料集上訓練 GPT 生成中文名字。

---

**上一篇**：[Transformer.md](Transformer.md)

**相關連結**：[Attention-Mechanism.md](Attention-Mechanism.md) | [RMSNorm.md](RMSNorm.md)