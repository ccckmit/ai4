# nn/datasets.md - 資料載入理論

本模組實現了資料載入的基礎設施：`Dataset` 與 `DataLoader` 抽象類別，以及 MNIST 資料集的載入和預處理工具。此模組需要 PyTorch 才能執行（`try/except ImportError` 保護）。

## Dataset 與 DataLoader 設計模式

### Dataset（資料集）

`Dataset` 是資料的抽象介面，定義了兩個方法：

```python
class Dataset:
    def __getitem__(self, index):  # 返回第 index 個樣本
    def __len__(self):             # 返回資料總數
```

這種設計支援**惰性載入（lazy loading）**：只在需要時才從硬碟讀取資料，適合大型資料集。

### DataLoader（資料載入器）

`DataLoader` 包裝一個 Dataset，提供批次迭代功能：

```python
loader = DataLoader(dataset, batch_size=32, shuffle=True)
for batch_x, batch_y in loader:
    logits = model(batch_x)
    loss = loss_fn(logits, batch_y)
```

核心功能：
1. **批次（Batching）**：將資料分為固定大小的群組
2. **打亂（Shuffling）**：每次 epoch 重新排列樣本順序
3. **迭代器協定**：實作 `__iter__` 支援 Python for 迴圈

### 為什麼需要 DataLoader

- **記憶體管理**：無法一次載入所有資料時，分批次處理
- **隨機梯度下降（SGD）**：小批次更新比全批次收斂更快、泛化更好
- **資料增強**：可以在載入時動態轉換資料

## MNIST（手寫數字資料集）

MNIST 是機器學習最經典的入門資料集，包含 70,000 張 28×28 的手寫數字灰階影像。

### MNIST 載入

`load_mnist()` 回傳 MNIST 的 NumPy 陣列：

```python
X_train, y_train, X_test, y_test = load_mnist()
# X_train.shape == (60000, 28, 28)
# y_train.shape == (60000,)
```

### 資料預處理轉換

`Compose` 類別實現了**管道模式（Pipeline Pattern）**，將多個轉換串聯：

```python
transforms = Compose([
    Grayscale(),
    Resize((28, 28)),
    ToTensor(),
    Normalize(mean=0.1307, std=0.3081),
])
```

各轉換的作用：
- **Grayscale**：確保單通道灰階
- **Resize**：統一影像尺寸
- **ToTensor**：NumPy 陣列 → 張量，並將數值縮放到 $[0, 1]$
- **Normalize**：標準化，$x = (x - mean) / std$

MNIST 的標準化參數 `mean=0.1307, std=0.3081` 是從整個訓練集計算得來的固定統計量。

### 隨機增強（訓練用）

```python
transforms = Compose([
    RandomRotation(degrees=10),  # 隨機旋轉 ±10 度
    ToTensor(),
    Normalize(mean=0.1307, std=0.3081),
])
```

資料增強能有效提升模型泛化能力，因為每 epoch 看到的資料都有微小的隨機變化。

詳細理論請見 [_wiki/DataLoader-Dataset.md](../_wiki/DataLoader-Dataset.md)。

---

**相關連結**：[cnn.md](cnn.md) | [nn.md](nn.md) | [tensor.md](tensor.md)
