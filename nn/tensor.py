"""
Tensor with automatic differentiation (autograd) based on NumPy.
Records operation history to support backpropagation through a computation graph.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Callable, Sequence


def unbroadcast(grad: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """
    Restores broadcast gradients to their original shape.
    When broadcasting expands dimensions, gradients must be summed over
    those dimensions during backpropagation.
    """
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
    NumPy-based tensor with autograd.
    Tracks computation history for backpropagation.

    Attributes:
        data: NumPy array holding the tensor values
        grad: Accumulated gradients (same shape as data)
        _backward: Function to call during backprop
        _prev: Set of parent tensors (for topological sort)
        requires_grad: Whether to track gradients for this tensor
    """

    def __init__(
        self,
        data: Any,
        _children: tuple[Tensor, ...] = (),
        requires_grad: bool = False,
    ) -> None:
        self.data: np.ndarray = np.array(data, dtype=np.float32)
        self.grad: np.ndarray = np.zeros_like(self.data)
        self._backward: Callable[[], None] = lambda: None
        self._prev: set[Tensor] = set(_children)
        self.requires_grad: bool = requires_grad

    @property
    def shape(self) -> tuple[int, ...]:
        """Returns the shape of the underlying NumPy array."""
        return self.data.shape

    def zero_grad(self) -> None:
        """Resets gradients to zeros."""
        self.grad = np.zeros_like(self.data)

    def backward(self) -> None:
        """
        Runs backpropagation from this tensor.
        Uses topological sort (逆序) to process nodes in correct order.
        Initializes gradient of this tensor to ones (dL/dL = 1).
        """
        topo: list[Tensor] = []
        visited: set[Tensor] = set()

        def build_topo(v: Tensor) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        self.grad = np.ones_like(self.data)
        for v in reversed(topo):
            v._backward()

    def __add__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward() -> None:
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)

        out._backward = _backward
        return out

    def __mul__(self, other: Tensor | float) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward() -> None:
            self.grad += unbroadcast(out.grad * other.data, self.data.shape)
            other.grad += unbroadcast(out.grad * self.data, other.data.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: Tensor) -> Tensor:
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data @ other.data, (self, other), self.requires_grad or other.requires_grad)

        def _backward() -> None:
            self.grad += unbroadcast(out.grad @ np.swapaxes(other.data, -1, -2), self.data.shape)
            other.grad += unbroadcast(np.swapaxes(self.data, -1, -2) @ out.grad, other.data.shape)

        out._backward = _backward
        return out

    def transpose(self, ax1: int, ax2: int) -> Tensor:
        """Swaps two axes. Gradient: swap the gradient axes back."""
        out = Tensor(np.swapaxes(self.data, ax1, ax2), (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += np.swapaxes(out.grad, ax1, ax2)

        out._backward = _backward
        return out

    def reshape(self, *shape: int) -> Tensor:
        """Reshapes tensor. Gradient: reshape gradient back to original shape."""
        out = Tensor(self.data.reshape(*shape), (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        """ReLU: max(0, x). Gradient is 1 where input > 0, else 0."""
        out = Tensor(np.maximum(0, self.data), (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += out.grad * (self.data > 0)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        """Tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"""
        out = Tensor(np.tanh(self.data), (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += out.grad * (1 - out.data ** 2)

        out._backward = _backward
        return out

    def clamp(self, min_val: float | None = None, max_val: float | None = None) -> Tensor:
        """Clamp tensor values to [min, max] range.

        FIX: original code only applied the mask when BOTH min_val and max_val
        were provided. One-sided clamps (e.g. clamp(min_val=0)) incorrectly
        passed gradients through clamped positions.
        Now each boundary is checked independently.
        """
        out = Tensor(np.clip(self.data, min_val, max_val), (self,), self.requires_grad)

        def _backward() -> None:
            mask = np.ones_like(self.data, dtype=bool)
            if min_val is not None:
                mask &= (self.data >= min_val)
            if max_val is not None:
                mask &= (self.data <= max_val)
            self.grad += out.grad * mask

        out._backward = _backward
        return out

    def abs(self) -> Tensor:
        """Absolute value."""
        out = Tensor(np.abs(self.data), (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += out.grad * np.sign(self.data)

        out._backward = _backward
        return out

    def masked_fill(self, mask: np.ndarray, value: float) -> Tensor:
        """Replaces values where mask is True with value. Gradient: mask out gradient at True positions."""
        out_data = np.where(mask, value, self.data)
        out = Tensor(out_data, (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += np.where(mask, 0, out.grad)

        out._backward = _backward
        return out

    def softmax(self, axis: int = -1) -> Tensor:
        """
        Softmax along given axis.
        Gradient: diag(softmax) - softmax @ softmax^T (Jacobian of softmax).
        """
        max_val = np.max(self.data, axis=axis, keepdims=True)
        exps = np.exp(self.data - max_val)
        probs = exps / np.sum(exps, axis=axis, keepdims=True)
        out = Tensor(probs, (self,), self.requires_grad)

        def _backward() -> None:
            s = out.data
            grad_s = out.grad
            self.grad += s * (grad_s - np.sum(grad_s * s, axis=axis, keepdims=True))

        out._backward = _backward
        return out

    def cross_entropy(self, targets: Tensor | np.ndarray) -> Tensor:
        """
        Cross-entropy loss for classification.
        targets: integer labels of shape (batch, seq_len) or (batch,)
        Converts targets to int64 numpy array for indexing.
        """
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

        def _backward() -> None:
            d_logits = probs.copy()
            for b in range(batch_size):
                for t in range(seq_len):
                    d_logits[b, t, targets_data[b, t]] -= 1
            d_logits = d_logits / (batch_size * seq_len)
            self.grad += out.grad * d_logits

        out._backward = _backward
        return out

    def __sub__(self, other: Tensor | float) -> Tensor:
        return self + (other * -1)

    def __truediv__(self, other: float) -> Tensor:
        return self * (other ** -1)

    def __radd__(self, other: float) -> Tensor:
        return self + other

    def __rmul__(self, other: float) -> Tensor:
        return self * other

    def __pow__(self, power: float) -> Tensor:
        out = Tensor(self.data ** power, (self,), self.requires_grad)

        def _backward() -> None:
            self.grad += out.grad * (power * self.data ** (power - 1))

        out._backward = _backward
        return out

    def sum(self, axis: int | None = None, keepdims: bool = False) -> Tensor:
        """Sum over specified axis. Gradient: broadcast gradient back.

        FIX: original code used `np.ones_like(self.data) * out.grad` which
        crashes when axis reduces a dimension (shape mismatch).
        Now restores the reduced dimension with expand_dims before broadcasting.
        """
        out = Tensor(self.data.sum(axis=axis, keepdims=keepdims), (self,), self.requires_grad)

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            self.grad += np.broadcast_to(grad, self.data.shape)

        out._backward = _backward
        return out

    def mean(self, axis: int | tuple[int, ...] | None = None, keepdims: bool = False) -> Tensor:
        """Mean over specified axis.

        FIX: same shape-broadcast issue as sum. Restores reduced dimension
        before broadcasting, then divides by the number of averaged elements.
        """
        out = Tensor(self.data.mean(axis=axis, keepdims=keepdims), (self,), self.requires_grad)

        def _backward() -> None:
            grad = out.grad
            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)
            if axis is None:
                n = self.data.size
            else:
                n = self.data.shape[axis] if isinstance(axis, int) else np.prod([self.data.shape[a] for a in axis])
            self.grad += np.broadcast_to(grad, self.data.shape) / n

        out._backward = _backward
        return out

    def __neg__(self) -> Tensor:
        return self * -1


def cat(tensors: Sequence[Tensor], axis: int = 0) -> Tensor:
    """
    Concatenates tensors along given axis.
    Backward: splits gradient along concatenation axis, distributes to original tensors.
    Used for KV Cache concatenation in Transformer attention.
    """
    data = np.concatenate([t.data for t in tensors], axis=axis)
    requires_grad = any(t.requires_grad for t in tensors)
    out = Tensor(data, tuple(tensors), requires_grad)

    def _backward() -> None:
        if not out.requires_grad:
            return
        split_sizes = [t.data.shape[axis] for t in tensors]
        grads = np.split(out.grad, np.cumsum(split_sizes)[:-1], axis=axis)
        for t, g in zip(tensors, grads):
            if t.requires_grad:
                t.grad += g

    out._backward = _backward
    return out