# Tensor（張量）與自動微分

張量是深度學習中的基本資料結構，廣義上是指多維陣列。本專案 `nn/tensor.py` 實現了一個簡化的張量類別，支援自動微分（automatic differentiation），讓用戶能以 NumPy 的語法操作神經網路，同時追蹤運算歷史用於反向傳播。

## 張量的資料表示

```python
self.data    # NumPy 陣列，儲存實際數值
self.grad    # NumPy 陣列，儲存梯度
self._backward  # 函數，計算本地梯度並傳給上游
self._prev      # set，記錄依賴的父張量（計算圖節點）
self.requires_grad  # bool，是否需要追蹤梯度
```

一個張量在建立時可以選擇是否追蹤梯度。設定 `requires_grad=True` 後，所有基於該張量的運算都會記錄到計算圖中，支援後續的梯度計算。

## 計算圖與前向傳播

每次對 Tensor 物件呼叫運算（如 `+`、`*`、`@`），會：
1. 執行對應的 NumPy 運算產生新的 Tensor（output）
2. 將 input Tensor 加入 output 的 `_prev` set（建立依賴關係）
3. 為 output 綁定一個 `_backward` 函數（稍後用於梯度計算）

例如，`c = a + b` 時：
```python
out = Tensor(a.data + b.data, (a, b), requires_grad=True)
def _backward():
    a.grad += out.grad  # 上游梯度傳給 a
    b.grad += out.grad  # 上游梯度傳給 b
out._backward = _backward
```

這就形成了計算圖：a 和 b 是 c 的上游節點。

## 反向傳播演算法

呼叫 `tensor.backward()` 時，會執行以下步驟：

1. **拓撲排序（Topological Sort）**：建立從 root（當前張量）到所有葉節點的計算順序
   
2. **初始化 root 梯度**：root 的梯度設為 `np.ones_like(data)`

3. **逆序執行 `_backward`**：按照拓撲排序的逆序，每個張量呼叫其 `_backward` 函數，將梯度累加到 `_prev` 節點

```python
def backward(self):
    topo = []
    visited = set()
    def build_topo(v):
        if v not in visited:
            visited.add(v)
            for child in v._prev:
                build_topo(child)
            topo.append(v)
    build_topo(self)
    
    self.grad = np.ones_like(self.data)
    for v in reversed(topo):
        v._backward()
```

這保證了每個節點在執行反向傳播時，其所有下游節點的梯度已經計算完成。

## 廣播與梯度還原

NumPy 支援廣播——不同形狀的陣列可以進行運算。例如：
```python
a.shape = (3, 1)  # 3×1
b.shape = (1, 4)  # 1×4
c = a + b         # 3×4 (廣播後)
```

廣播後的梯度需要「還原」到廣播前的形狀。本專案 `unbroadcast(grad, shape)` 函數實現了這個邏輯：
- 若梯度比目標形狀多前導維度，在這些維度上求和
- 若某維度原本是 1 但現在變大，在該維度上求和

## 矩陣乘法的梯度

對於矩陣乘法 $C = AB$，梯度反向傳播時：
$$\frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot B^T$$
$$\frac{\partial L}{\partial B} = A^T \cdot \frac{\partial L}{\partial C}$$

本專案的實現考慮了 transpose 交換軸的影響，使用 `np.swapaxes` 處理高維矩陣乘法的梯度。

## 支持的運算

### 基礎算術
- `+`, `*`, `-`, `/`（透過魔法方法實現）
- `@` 矩陣乘法
- `**` 冪運算
- 一元 `-`（negation）

### 激活函數
- `relu()`：$\max(0, x)$，梯度為 `x > 0`
- `softmax(axis)`：歸一化指數，梯度為 $p(\delta_{ij} - p_j)$
- `masked_fill(mask, value)`：根據條件替換值

### 形狀操作
- `transpose(ax1, ax2)`：交換兩個軸
- `reshape(*shape)`：改變形狀
- `sum(axis, keepdims)`：求和

### 損失函數
- `cross_entropy(targets)`：分類任務常用損失

## 與 PyTorch 的比較

本專案的 Tensor 類別借鑒了 PyTorch 的設計理念，但做了極度簡化：

| 特性 | PyTorch | 本專案 |
|------|---------|--------|
| 資料結構 | CUDA Tensor | NumPy array |
| 計算圖 | 動態圖（每次前向動態建構） | 簡化的前向記錄 |
| GPU 支援 | 是 | 否 |
| 梯度累加 | 預設不覆蓋 | 當前實現為 += |
| 記憶體效率 | 精細管理 | 簡化 |

本專案適合學習自動微分原理、生產簡單模型原型。實務訓練建議使用 PyTorch 或 JAX。

## 設計取捨

1. **梯度累加方式**：PyTorch 中葉節點的梯度是累加的（`+=`），本專案也採用同樣策略。這在多次 backward 時有影響。
2. **無梯度追蹤控制**：本專案沒有 `detach()` 之類的方法，但可以透過不使用 `requires_grad=True` 來避免追蹤。
3. **記憶體管理**：本專案不做原地操作，每個運算都產生新 Tensor，記憶體效率較低但實現簡單。

---

**相關連結**：[Backpropagation.md](../_wiki/Backpropagation.md) | [nn/optim.md](../nn/optim.md)