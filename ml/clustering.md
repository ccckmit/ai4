# ml/clustering.md - 聚類理論

本模組實現了兩種非監督式學習演算法：**K-Means** 和 **DBSCAN**。聚類（Clustering）的目標是將資料點分組，讓同一組內的點相似度高，不同組間的相似度低。

## K-Means

K-Means 是最經典的聚類演算法，透過反覆迭代來找到 K 個聚類中心。

### 演算法流程

1. **初始化**：隨機選取 K 個資料點作為初始中心
2. **分配步驟**：每個點分配到最近的中心
3. **更新步驟**：重新計算每個聚類的平均值作為新中心
4. **重複 2-3** 直到收斂（中心不再變化）

### 距離度量與目標函數

K-Means 最小化**慣量（Inertia）**，即各點到其聚類中心的平方距離：

$$\mathcal{L} = \sum_{i=1}^N \sum_{k=1}^K r_{ik} \|x_i - \mu_k\|^2$$

其中 $r_{ik} = 1$ 若點 $i$ 分配到聚類 $k$，否則為 0。

### K 值的選擇

K-Means 需要預先指定 K，這是它最大的限制。常用選擇方法：

- **肘部法則（Elbow Method）**：繪製 K vs inertia 曲線，找拐點
- **輪廓係數（Silhouette Score）**：衡量聚類的緊湊度與分離度

### 初始化方法

隨機初始化可能導致收斂到局部最佳解。K-Means++ 是一種改進：
1. 隨機選第一個中心
2. 後續中心以機率 $\frac{D(x)^2}{\sum D(x)^2}$ 選取（$D(x)$ 是到最近已選中心的距離）
3. 這保證初始中心分布較均勻

## DBSCAN

DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是基於密度的聚類演算法，不需要預先指定群數。

### 核心概念

- **核心點（Core Point）**：在 $\epsilon$ 半徑內至少有 `min_samples` 個點
- **邊界點（Border Point）**：在核心點的 $\epsilon$ 半徑內，但不是核心點
- **噪聲點（Noise Point）**：既非核心也非邊界

### 演算法

1. 對每個點，找出 $\epsilon$ 半徑內的鄰居
2. 標記核心點（鄰居數 $\ge$ min_samples）
3. 核心點之間若相連則屬於同一聚類
4. 邊界點歸屬於它所在的核心聚類
5. 其餘為噪聲

### K-Means vs DBSCAN

| 特性 | K-Means | DBSCAN |
|------|---------|--------|
| 群數 | 需指定 K | 自動發現 |
| 形狀 | 球形聚類 | 任意形狀 |
| 噪聲處理 | 不支援（所有點都分配） | 支援（標記為噪聲） |
| 密度不均 | 效果差 | 效果差 |
| 高維資料 | 可運作 | 距離計算退化 |

詳細理論請見 [_wiki/KMeans.md](../_wiki/KMeans.md)。

---

**相關連結**：[decomposition.md](decomposition.md) | [preprocessing.md](preprocessing.md)
