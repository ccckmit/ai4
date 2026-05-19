import numpy as np

"""
  處理 NumPy 廣播機制在反向傳播時的梯度還原。
  當張量形狀因廣播而不同時，需要在擴展的維度上求和。
"""
def unbroadcast(grad, shape):
    if grad.shape == shape:
        return grad
    ndim_diff = grad.ndim - len(shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))
    for i, dim in enumerate(shape):
        if dim == 1 and grad.shape[i] > 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad

class Tensor:
    """
    自動微分張量（基於 NumPy）。
    記錄運算歷史並透過計算圖支援反向傳播。
    """
    def __init__(self, data, _children=(), requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self.requires_grad = requires_grad

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

    """反向傳播：拓樸排序後逆序執行每個節點的 _backward"""
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

    # --- 基礎運算 ---
    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)
        out._backward = _backward
        return out

    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), self.requires_grad or other.requires_grad)
        def _backward():
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape)
        out._backward = _backward
        return out

    # --- 張量形狀操作 ---
    def transpose(self, ax1, ax2):
        out = Tensor(np.swapaxes(self.data, ax1, ax2), (self,), self.requires_grad)
        def _backward():
            self.grad += np.swapaxes(out.grad, ax1, ax2)
        out._backward = _backward
        return out

    def reshape(self, *shape):
        out = Tensor(self.data.reshape(*shape), (self,), self.requires_grad)
        def _backward():
            self.grad += out.grad.reshape(self.data.shape)
        out._backward = _backward
        return out

    # --- 激活與非線性函數 ---
    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), self.requires_grad)
        def _backward():
            self.grad += out.grad * (self.data > 0)
        out._backward = _backward
        return out

    def masked_fill(self, mask, value):
        out_data = np.where(mask, value, self.data)
        out = Tensor(out_data, (self,), self.requires_grad)
        def _backward():
            self.grad += np.where(mask, 0, out.grad)
        out._backward = _backward
        return out

    def softmax(self, axis=-1):
        max_val = np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(self.data - max_val)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, (self,), self.requires_grad)
        def _backward():
            s = out.data
            grad_s = out.grad
            self.grad += s * (grad_s - np.sum(grad_s * s, axis=axis, keepdims=True))
        out._backward = _backward
        return out

    def cross_entropy(self, targets):
        targets_data = np.asarray(targets.data if isinstance(targets, Tensor) else targets, dtype=np.int64)
        logits = self.data
        max_logits = np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(logits - max_logits)
        probs = exps / np.sum(exps, axis=-1, keepdims=True)

        batch_size, seq_len = targets_data.shape
        loss = 0.0
        for b in range(batch_size):
            for t in range(seq_len):
                loss -= np.log(probs[b, t, targets_data[b, t]] + 1e-10)
        loss = loss / (batch_size * seq_len)

        out = Tensor(loss, (self,), self.requires_grad)
        def _backward():
            d_logits = probs.copy()
            for b in range(batch_size):
                for t in range(seq_len):
                    d_logits[b, t, targets_data[b, t]] -= 1
            d_logits = d_logits / (batch_size * seq_len)
            self.grad += out.grad * d_logits
        out._backward = _backward
        return out

    # Python Magic Methods
    def __sub__(self, other): return self + (other * -1)
    def __truediv__(self, other): return self * (other ** -1)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __pow__(self, power):
        out = Tensor(self.data ** power, (self,), self.requires_grad)
        def _backward():
            self.grad += out.grad * (power * self.data ** (power - 1))
        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), self.requires_grad)
        def _backward():
            self.grad += np.ones_like(self.data) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1

"""
  cat：將多個 Tensor 沿指定維度拼接。
  反向傳播時將梯度沿拼接維度切分，分配給原始張量。
  用於 KV Cache 的 Key/Value 拼接操作。
"""
def cat(tensors, axis=0):
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, tuple(tensors), requires_grad)
    
    def _backward():
        if not out.requires_grad: return
        split_sizes = [t.data.shape[axis] for t in tensors]
        grads = np.split(out.grad, np.cumsum(split_sizes)[:-1], axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g
    out._backward = _backward
    return out