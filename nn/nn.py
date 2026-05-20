"""
Neural network layers and optimizers.
Module: base class for all neural network components
Linear: fully connected layer (y = x @ W + b)
Embedding: token embedding lookup table
RMSNorm: Root Mean Square Normalization
Adam: Adaptive Moment Estimation optimizer
"""

from __future__ import annotations

import numpy as np
from .tensor import Tensor


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    """Mean Squared Error loss.

    FIX: original _backward read diff.grad, which is always zero because
    diff = input - target is an intermediate node created outside the
    autograd graph registered to `out`. backward() never traverses diff.
    Now computes the gradient directly from diff.data (the stored values).
    """
    diff = input - target
    out = Tensor(np.mean(diff.data ** 2), (input, target), input.requires_grad or target.requires_grad)

    def _backward() -> None:
        N = np.prod(diff.data.shape)
        input.grad  += out.grad * 2 * diff.data / N
        target.grad += out.grad * -2 * diff.data / N

    out._backward = _backward
    return out


class Module:
    """
    Base class for all neural network modules.
    Provides parameter collection via recursive traversal.
    """

    def parameters(self) -> list[Tensor]:
        """
        Recursively collects all Tensors with requires_grad=True.
        Traverses Module attributes, lists of Modules, and direct Tensors.
        """
        params: list[Tensor] = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params


class Sequential(Module):
    """Sequential container - applies modules in order."""

    def __init__(self, *modules: Module) -> None:
        self.modules: tuple[Module, ...] = modules

    def __call__(self, x: Tensor) -> Tensor:
        for m in self.modules:
            x = m(x)
        return x


class ReLU(Module):
    """ReLU activation layer."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.relu()


class Tanh(Module):
    """Tanh activation layer."""

    def __call__(self, x: Tensor) -> Tensor:
        return x.tanh()


class Linear(Module):
    """
    Fully connected linear transformation: y = x @ W + b
    Weight shape: (in_features, out_features)
    Bias shape: (out_features,) if bias=True, else None
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False) -> None:
        std = 0.08
        self.weight = Tensor(np.random.normal(0, std, (in_features, out_features)), requires_grad=True)
        self.bias: Tensor | None = Tensor(np.zeros((out_features,)), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out


class Embedding(Module):
    """
    Token embedding lookup table.
    Maps integer indices to embedding vectors.
    Gradient accumulation uses np.add.at to handle duplicate indices correctly.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int) -> None:
        self.weight = Tensor(np.random.normal(0, 0.08, (num_embeddings, embedding_dim)), requires_grad=True)

    def __call__(self, indices: Tensor) -> Tensor:
        idx = indices.data.astype(int) if isinstance(indices, Tensor) else indices
        out_data = self.weight.data[idx]
        out = Tensor(out_data, (self.weight,), requires_grad=True)

        def _backward() -> None:
            np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out


class RMSNorm(Module):
    """
    Root Mean Square Normalization: x / RMS(x)
    RMS(x) = sqrt(mean(x^2) + eps)
    Cheaper than LayerNorm (no mean subtraction), used in LLaMA and modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        self.eps: float = eps
        self.scale = Tensor(np.ones(dim), requires_grad=False)

    def __call__(self, x: Tensor) -> Tensor:
        ms = np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps
        inv_std = ms ** -0.5
        out_data = x.data * inv_std
        out = Tensor(out_data, (x,), requires_grad=True)

        def _backward() -> None:
            N = x.data.shape[-1]
            dx = (out.grad * inv_std) - (
                x.data * inv_std ** 3 * np.sum(out.grad * x.data, axis=-1, keepdims=True) / N
            )
            x.grad += dx

        out._backward = _backward
        return out


class Adam(Module):
    """
    Adam optimizer (Adaptive Moment Estimation).

    Maintains two exponential moving averages per parameter:
    - m: first moment (gradient momentum), biased toward zero
    - v: second moment (squared gradient momentum), biased toward zero

    Update rule with bias correction:
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: list[Tensor],
        lr: float = 0.01,
        betas: tuple[float, float] = (0.85, 0.99),
        eps: float = 1e-8,
    ) -> None:
        self.params: list[Tensor] = params
        self.lr: float = lr
        self.beta1: float
        self.beta2: float
        self.beta1, self.beta2 = betas
        self.eps: float = eps
        self.m: list[np.ndarray] = [np.zeros_like(p.data) for p in params]
        self.v: list[np.ndarray] = [np.zeros_like(p.data) for p in params]
        self.t: int = 0

    def step(self) -> None:
        """Performs one optimization step."""
        self.t += 1
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self) -> None:
        """Resets gradients of all parameters."""
        for p in self.params:
            p.zero_grad()