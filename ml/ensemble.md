# ml/ensemble.md - 集成學習理論

本模組實現了兩種集成學習（Ensemble Learning）演算法：**隨機森林（Random Forest）** 和 **梯度提升（Gradient Boosting）**。集成學習的核心思想是：多個弱學習器組合可以形成一個強學習器。

## 隨機森林（Random Forest）

隨機森林是 Bagging（Bootstrap Aggregating）的代表性演算法，在決策樹的基礎上引入了**雙重隨機性**。

### Bagging 原理

1. 從原始資料集**有放回**抽樣 B 個子集（bootstrap samples）
2. 每個子集訓練一棵決策樹
3. 預測時取所有樹的投票（分類）或平均（回歸）

### 隨機森林的雙重隨機性

隨機森林在 Bagging 的基礎上增加**特徵隨機選擇**：

1. **樣本隨機性**：每棵樹使用不同的 bootstrap sample
2. **特徵隨機性**：每個節點分割時，只考慮 $\sqrt{n\_features}$ 或 $\log_2(n\_features)$ 個隨機選取的特徵

這保證了樹之間的多樣性。若所有樹都用相同的特徵，結果會高度相關，集成效果大打折扣。

### 預測方式

```python
# 分類：多數投票
predictions = [tree.predict(X) for tree in self.trees]
y_pred = mode(predictions, axis=0)

# 回歸：平均
y_pred = mean(predictions, axis=0)
```

### 隨機森林的優點

- **抗過擬合**：比單棵決策樹更穩定
- **特徵重要性**：可以計算每個特徵在分割中的貢獻度
- **平行化**：每棵樹可以獨立訓練
- **不需要太多調參**：預設超參數通常表現不錯

## 梯度提升（Gradient Boosting）

梯度提升是 Boosting 方法的代表，以**逐步修正殘差**的方式建立模型。

### 核心思想

每一棵新樹學習的是前一棵樹的**殘差（residual）**：

1. 初始模型預測為常數值：$F_0(x) = \arg\min_\gamma \sum L(y_i, \gamma)$
2. 對 $m = 1$ 到 $M$：
   - 計算殘差：$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F_{m-1}}$
   - 用決策樹擬合殘差
   - $F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$

其中 $\eta$（learning rate）控制每棵樹的貢獻程度。

### 與 Random Forest 的比較

| 特性 | Random Forest | Gradient Boosting |
|------|-------------|-------------------|
| 訓練方式 | 並行 | 序列 |
| 偏差 | 較低（Bagging 減少變異數） | 逐步降低偏差 |
| 變異數 | 較低 | 可能較高 |
| 對異常值敏感度 | 低 | 高 |
| 調參難度 | 低 | 高（需控制學習率） |

### 本專案實現

本模組實作了 `RandomForestClassifier`、`RandomForestRegressor`、`GradientBoostingClassifier`。其中 Gradient Boosting 使用決策樹作為基礎學習器，透過 learning rate 控制每棵樹的影響力。

詳細理論請見 [_wiki/Random-Forest.md](../_wiki/Random-Forest.md)。

---

**相關連結**：[tree.md](tree.md) | [linear_models.md](linear_models.md)
