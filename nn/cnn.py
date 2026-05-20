"""
Convolutional Neural Network layers.
Conv2d: 2D convolution layer
MaxPool2d: 2D max pooling
AvgPool2d: 2D average pooling
Flatten: flatten tensor for dense layers
"""

from __future__ import annotations

import numpy as np
from .tensor import Tensor
from .nn import Module


class Conv2d(Module):
    """
    2D convolution layer.

    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel (square kernel)
        stride: Stride of the convolution (default 1)
        padding: Padding added to input (default 0)
        bias: If True, adds a learnable bias (default True)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.in_channels: int = in_channels
        self.out_channels: int = out_channels
        self.kernel_size: int = kernel_size
        self.stride: int = stride
        self.padding: int = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(
            np.random.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.bias: Tensor | None = (
            Tensor(np.zeros(out_channels), requires_grad=True) if bias else None
        )

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of 2D convolution."""
        batch_size, in_channels, height, width = x.data.shape

        out_height = (height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (width + 2 * self.padding - self.kernel_size) // self.stride + 1

        if self.padding > 0:
            padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode="constant",
            )
        else:
            padded = x.data

        out = np.zeros((batch_size, self.out_channels, out_height, out_width), dtype=np.float32)

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for ic in range(self.in_channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            ih = oh * self.stride
                            iw = ow * self.stride
                            kernel = padded[
                                b,
                                ic,
                                ih : ih + self.kernel_size,
                                iw : iw + self.kernel_size,
                            ]
                            out[b, oc, oh, ow] += np.sum(
                                kernel * self.weight.data[oc, ic]
                            )

                if self.bias is not None:
                    out[b, oc] += self.bias.data[oc]

        result = Tensor(out, (x, self.weight), x.requires_grad or self.weight.requires_grad)

        def _backward() -> None:
            grad = result.grad
            for b in range(batch_size):
                for oc in range(self.out_channels):
                    for ic in range(self.in_channels):
                        for oh in range(out_height):
                            for ow in range(out_width):
                                ih = oh * self.stride
                                iw = ow * self.stride
                                x.grad[b, ic, ih : ih + self.kernel_size, iw : iw + self.kernel_size] += (
                                    grad[b, oc, oh, ow] * self.weight.data[oc, ic]
                                )
                                self.weight.grad[oc, ic] += (
                                    grad[b, oc, oh, ow]
                                    * padded[
                                        b,
                                        ic,
                                        ih : ih + self.kernel_size,
                                        iw : iw + self.kernel_size,
                                    ]
                                )
                    if self.bias is not None:
                        self.bias.grad[oc] += np.sum(grad[b, oc])

        result._backward = _backward
        return result


class MaxPool2d(Module):
    """
    2D max pooling layer.

    Args:
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (default equal to kernel_size)
    """

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        self.kernel_size: int = kernel_size
        self.stride: int = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of 2D max pooling."""
        batch_size, channels, height, width = x.data.shape

        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)
        indices = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=np.int32)

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        ih = oh * self.stride
                        iw = ow * self.stride
                        window = x.data[
                            b, c, ih : ih + self.kernel_size, iw : iw + self.kernel_size
                        ]
                        max_val = np.max(window)
                        out[b, c, oh, ow] = max_val

                        max_idx = np.unravel_index(np.argmax(window), window.shape)
                        indices[b, c, oh, ow] = [ih + max_idx[0], iw + max_idx[1]]

        result = Tensor(out, (x,), x.requires_grad)
        result._pool_indices = indices
        result._input_shape = x.data.shape

        def _backward() -> None:
            grad = result.grad
            for b in range(batch_size):
                for c in range(channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            ih, iw = indices[b, c, oh, ow]
                            x.grad[b, c, ih, iw] += grad[b, c, oh, ow]

        result._backward = _backward
        return result


class AvgPool2d(Module):
    """
    2D average pooling layer.

    Args:
        kernel_size: Size of the pooling window
        stride: Stride of the pooling (default equal to kernel_size)
    """

    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        self.kernel_size: int = kernel_size
        self.stride: int = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of 2D average pooling."""
        batch_size, channels, height, width = x.data.shape

        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1

        out = np.zeros((batch_size, channels, out_height, out_width), dtype=np.float32)

        for b in range(batch_size):
            for c in range(channels):
                for oh in range(out_height):
                    for ow in range(out_width):
                        ih = oh * self.stride
                        iw = ow * self.stride
                        window = x.data[
                            b, c, ih : ih + self.kernel_size, iw : iw + self.kernel_size
                        ]
                        out[b, c, oh, ow] = np.mean(window)

        result = Tensor(out, (x,), x.requires_grad)

        def _backward() -> None:
            grad = result.grad
            pool_size = self.kernel_size * self.kernel_size
            for b in range(batch_size):
                for c in range(channels):
                    for oh in range(out_height):
                        for ow in range(out_width):
                            ih = oh * self.stride
                            iw = ow * self.stride
                            x.grad[b, c, ih : ih + self.kernel_size, iw : iw + self.kernel_size] += (
                                grad[b, c, oh, ow] / pool_size
                            )

        result._backward = _backward
        return result


class Flatten(Module):
    """Flatten tensor from (B, C, H, W) to (B, C*H*W)."""

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Flatten input tensor."""
        batch_size = x.data.shape[0]
        new_shape = (batch_size, np.prod(x.data.shape[1:]))
        out = Tensor(x.data.reshape(new_shape), (x,), x.requires_grad)

        def _backward() -> None:
            x.grad += out.grad.reshape(x.data.shape)

        out._backward = _backward
        return out


class BatchNorm2d(Module):
    """
    2D batch normalization layer.

    Args:
        num_channels: Number of channels
        eps: Epsilon for numerical stability
        momentum: Momentum for moving average
    """

    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.9,
    ) -> None:
        self.num_channels: int = num_channels
        self.eps: float = eps
        self.momentum: float = momentum

        self.weight = Tensor(np.ones(num_channels), requires_grad=True)
        self.bias = Tensor(np.zeros(num_channels), requires_grad=True)
        self.running_mean: np.ndarray = np.zeros(num_channels)
        self.running_var: np.ndarray = np.ones(num_channels)
        self.training: bool = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of batch normalization."""
        batch_size, channels, height, width = x.data.shape

        if self.training:
            mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x.data, axis=(0, 2, 3), keepdims=True)
        else:
            mean = self.running_mean.reshape(1, channels, 1, 1)
            var = self.running_var.reshape(1, channels, 1, 1)

        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        out_data = self.weight.data.reshape(1, channels, 1, 1) * x_norm + self.bias.data.reshape(
            1, channels, 1, 1
        )

        out = Tensor(out_data, (x, self.weight, self.bias), x.requires_grad)

        def _backward() -> None:
            x_grad = out.grad * self.weight.data.reshape(1, channels, 1, 1)
            self.weight.grad += np.sum(out.grad * x_norm, axis=(0, 2, 3))
            self.bias.grad += np.sum(out.grad, axis=(0, 2, 3))

            dx = x_grad / np.sqrt(var + self.eps)
            db = np.mean(dx, axis=(0, 2, 3), keepdims=True)
            x.grad += dx - db

        out._backward = _backward
        return out

    def eval(self) -> None:
        """Set to evaluation mode."""
        self.training = False

    def train(self) -> None:
        """Set to training mode."""
        self.training = True


class Dropout2d(Module):
    """
    2D dropout layer.

    Args:
        p: Probability of dropping a channel
    """

    def __init__(self, p: float = 0.5) -> None:
        self.p: float = p
        self.training: bool = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of 2D dropout."""
        if not self.training:
            return x

        mask = np.random.binomial(1, 1 - self.p, x.data.shape)
        out_data = x.data * mask / (1 - self.p)
        out = Tensor(out_data, (x,), x.requires_grad)

        def _backward() -> None:
            x.grad += out.grad * mask / (1 - self.p)

        out._backward = _backward
        return out

    def eval(self) -> None:
        """Set to evaluation mode."""
        self.training = False

    def train(self) -> None:
        """Set to training mode."""
        self.training = True