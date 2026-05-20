"""
Neural network layers and optimizers.
Module: base class for all neural network components
Linear: fully connected layer (y = x @ W + b)
Embedding: token embedding lookup table
RMSNorm: Root Mean Square Normalization
Adam: Adaptive Moment Estimation optimizer
"""

import numpy as np
from .tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    Provides parameter collection via recursive traversal.
    """

    def parameters(self):
        """
        Recursively collects all Tensors with requires_grad=True.
        Traverses Module attributes, lists of Modules, and direct Tensors.
        """
        params = []
        for v in self.__dict__.values():
            if isinstance(v, Tensor) and v.requires_grad:
                params.append(v)
            elif isinstance(v, Module):
                params.extend(v.parameters())
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, Module):
                        params.extend(item.parameters())
        return params


class Linear(Module):
    """
    Fully connected linear transformation: y = x @ W + b
    Weight shape: (in_features, out_features)
    Bias shape: (out_features,) if bias=True, else None
    """

    def __init__(self, in_features, out_features, bias=False):
        std = 0.08
        self.weight = Tensor(np.random.normal(0, std, (in_features, out_features)), requires_grad=True)
        self.bias = Tensor(np.zeros((out_features,)), requires_grad=True) if bias else None

    def __call__(self, x):
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

    def __init__(self, num_embeddings, embedding_dim):
        self.weight = Tensor(np.random.normal(0, 0.08, (num_embeddings, embedding_dim)), requires_grad=True)

    def __call__(self, indices):
        # Support both Tensor and raw numpy array indices
        idx = indices.data.astype(int) if isinstance(indices, Tensor) else indices
        out_data = self.weight.data[idx]
        out = Tensor(out_data, (self.weight,), requires_grad=True)

        def _backward():
            np.add.at(self.weight.grad, idx, out.grad)

        out._backward = _backward
        return out


class RMSNorm(Module):
    """
    Root Mean Square Normalization: x / RMS(x)
    RMS(x) = sqrt(mean(x^2) + eps)
    Cheaper than LayerNorm (no mean subtraction), used in LLaMA and modern LLMs.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.scale = Tensor(np.ones(dim), requires_grad=False)

    def __call__(self, x):
        # RMS = sqrt(mean(x^2) + eps)
        ms = np.mean(x.data ** 2, axis=-1, keepdims=True) + self.eps
        inv_std = ms ** -0.5
        out_data = x.data * inv_std
        out = Tensor(out_data, (x,), requires_grad=True)

        def _backward():
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

    def __init__(self, params, lr=0.01, betas=(0.85, 0.99), eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [np.zeros_like(p.data) for p in params]  # First moment
        self.v = [np.zeros_like(p.data) for p in params]  # Second moment
        self.t = 0  # Timestep for bias correction

    def step(self):
        """Performs one optimization step."""
        self.t += 1
        for i, p in enumerate(self.params):
            # Update biased first moment estimate
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            # Update biased second moment estimate
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (p.grad ** 2)
            # Bias-corrected moments
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # Parameter update
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        """Resets gradients of all parameters."""
        for p in self.params:
            p.zero_grad()