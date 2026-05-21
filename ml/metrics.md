# ml/metrics.md - 評估指標理論

本模組實現了機器學習常用的評估指標，用於量化模型預測的品質。評估指標是模型選擇和調參的依據。

## 分類指標

### Accuracy（準確率）

最直觀的分類指標：預測正確的比例。

$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

- 適合**類別平衡**的資料集
- 類別不平衡時會失真（如 99% 負類別，全部猜負也有 99%）

### Confusion Matrix（混淆矩陣）

混淆矩陣是一個 $K \times K$ 的表格，$C_{ij}$ 表示真實類別 $i$ 被預測為類別 $j$ 的數量：

```
                  Predicted
              0        1
Actual 0   [[TN,      FP],
        1    [FN,      TP]]
```

從混淆矩陣可以衍生多種指標：
- **Precision**（精確率）：$TP / (TP + FP)$ — 所有被預測為正類別中，有多少是真的正類別
- **Recall**（召回率）：$TP / (TP + FN)$ — 所有真正的正類別中，有多少被正確識別
- **F1-Score**：$2 \cdot Precision \cdot Recall / (Precision + Recall)$ — Precision 和 Recall 的調和平均

## 回歸指標

### MSE（均方誤差）

$$MSE = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2$$

- 懲罰大誤差（平方效應）
- 單位是目標變數的平方，解釋性較差

### R²（決定係數）

$$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$

- 範圍 $(-\infty, 1]$，最理想為 1
- 解釋為「模型解釋了多少比例的變異數」
- $R^2 = 0$ 表示模型效果等同於直接用平均值預測
- $R^2 < 0$ 表示模型比平均值還差

### 何時使用哪個指標

| 場景 | 指標 | 原因 |
|------|------|------|
| 類別平衡分類 | Accuracy | 直觀，直接反映正確率 |
| 類別不平衡分類 | F1-Score / Confusion Matrix | 避免多數類別主導 |
| 需要可解釋的誤差 | RMSE（MSE 開根號） | 與原始資料單位相同 |
| 評估模型解釋力 | R² | 標準化，可跨任務比較 |

---

**相關連結**：[linear_models.md](linear_models.md) | [tree.md](tree.md) | [ensemble.md](ensemble.md)
