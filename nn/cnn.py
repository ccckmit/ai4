"""
Convolutional Neural Network layers.
Conv2d: 2D convolution layer
MaxPool2d: 2D max pooling
AvgPool2d: 2D average pooling
Flatten: flatten tensor for dense layers
BatchNorm2d: 2D Spatial Batch Normalization
Dropout2d: 2D Channel-wise Dropout
"""

from __future__ import annotations

import numpy as np
from .tensor import Tensor
from .nn import Module

# =====================================================================
# 輔助函數：im2col 與 col2im (用於將卷積/池化轉換為極速的矩陣相乘)
# =====================================================================

def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    N, C, H, W = x_shape
    out_height = (H + 2 * padding - field_height) // stride + 1
    out_width = (W + 2 * padding - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    return k, i, j

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')
    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)
    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]


# =====================================================================
# 神經網路層實作
# =====================================================================

class Conv2d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ) -> None:
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.weight = Tensor(
            np.random.normal(0, scale, (out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True,
        )
        self.bias = Tensor(np.zeros(out_channels), requires_grad=True) if bias else None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1

        # 1. 極速優化：將圖片轉換為欄位 (im2col)
        x_col = im2col_indices(x.data, self.kernel_size, self.kernel_size, self.padding, self.stride)
        
        # 2. 展開權重
        w_row = self.weight.data.reshape(self.out_channels, -1)
        
        # 3. 呼叫底層 C 做大矩陣相乘 (1 個 dot 抵銷原本的 5 層迴圈)
        out_col = w_row @ x_col
        if self.bias is not None:
            out_col += self.bias.data.reshape(-1, 1)
            
        # 4. 將結果塑形回 (N, out_channels, out_h, out_w)
        out_data = out_col.reshape(self.out_channels, out_h, out_w, N).transpose(3, 0, 1, 2)

        deps = (x, self.weight) if self.bias is None else (x, self.weight, self.bias)
        req_grad = x.requires_grad or self.weight.requires_grad or (self.bias and self.bias.requires_grad)
        result = Tensor(out_data, deps, req_grad)

        def _backward() -> None:
            dout = result.grad
            dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)

            # 計算權重梯度
            if self.weight.requires_grad:
                dW = dout_reshaped @ x_col.T
                self.weight.grad += dW.reshape(self.weight.data.shape)

            # 計算偏差梯度
            if self.bias is not None and self.bias.requires_grad:
                self.bias.grad += np.sum(dout_reshaped, axis=1)

            # 計算輸入梯度
            if x.requires_grad:
                dx_col = w_row.T @ dout_reshaped
                dx = col2im_indices(dx_col, x.data.shape, self.kernel_size, self.kernel_size, self.padding, self.stride)
                x.grad += dx

        result._backward = _backward
        return result


class MaxPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        # 將每個 Channel 視為獨立的 Batch 以共用 im2col
        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, self.kernel_size, self.kernel_size, padding=0, stride=self.stride)

        # 找出最大值與其索引
        max_idx = np.argmax(x_col, axis=0)
        out_col = x_col[max_idx, np.arange(x_col.shape[1])]
        out = out_col.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        result = Tensor(out, (x,), x.requires_grad)

        def _backward() -> None:
            dout = result.grad
            dout_col = dout.transpose(2, 3, 0, 1).ravel()
            
            # 建立 dx_col，只在最大值的位置放入梯度
            dx_col = np.zeros_like(x_col)
            dx_col[max_idx, np.arange(dx_col.shape[1])] = dout_col
            
            dx_reshaped = col2im_indices(dx_col, (N * C, 1, H, W), self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
            x.grad += dx_reshaped.reshape(N, C, H, W)

        result._backward = _backward
        return result


class AvgPool2d(Module):
    def __init__(self, kernel_size: int, stride: int | None = None) -> None:
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        out_h = (H - self.kernel_size) // self.stride + 1
        out_w = (W - self.kernel_size) // self.stride + 1

        x_reshaped = x.data.reshape(N * C, 1, H, W)
        x_col = im2col_indices(x_reshaped, self.kernel_size, self.kernel_size, padding=0, stride=self.stride)

        out_col = np.mean(x_col, axis=0)
        out = out_col.reshape(out_h, out_w, N, C).transpose(2, 3, 0, 1)

        result = Tensor(out, (x,), x.requires_grad)

        def _backward() -> None:
            dout = result.grad
            dout_col = dout.transpose(2, 3, 0, 1).ravel()
            
            # 將梯度均勻分佈到 kernel_size * kernel_size 個元素上
            dx_col = np.zeros_like(x_col)
            dx_col[:, :] = (dout_col / (self.kernel_size ** 2))[np.newaxis, :]
            
            dx_reshaped = col2im_indices(dx_col, (N * C, 1, H, W), self.kernel_size, self.kernel_size, padding=0, stride=self.stride)
            x.grad += dx_reshaped.reshape(N, C, H, W)

        result._backward = _backward
        return result


class Flatten(Module):
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        batch_size = x.data.shape[0]
        new_shape = (batch_size, -1)
        out = Tensor(x.data.reshape(new_shape), (x,), x.requires_grad)

        def _backward() -> None:
            x.grad += out.grad.reshape(x.data.shape)

        out._backward = _backward
        return out


class BatchNorm2d(Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        momentum: float = 0.1,  # PyTorch 預設 momentum 是 0.1
    ) -> None:
        self.num_channels = num_channels
        self.eps = eps
        self.momentum = momentum

        self.weight = Tensor(np.ones(num_channels), requires_grad=True)
        self.bias = Tensor(np.zeros(num_channels), requires_grad=True)
        self.running_mean = np.zeros(num_channels)
        self.running_var = np.ones(num_channels)
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape

        if self.training:
            # 計算目前 Batch 的 mean 和 var
            mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
            var = np.var(x.data, axis=(0, 2, 3), keepdims=True)
            
            # 修復：在訓練模式下更新移動平均，供 eval 模式使用
            m = mean.flatten()
            v = var.flatten()
            # PyTorch 的 EMA 公式為：new = (1-m) * new + m * old
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * m
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * v
        else:
            mean = self.running_mean.reshape(1, C, 1, 1)
            var = self.running_var.reshape(1, C, 1, 1)

        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_centered = x.data - mean
        x_norm = x_centered * std_inv
        
        gamma = self.weight.data.reshape(1, C, 1, 1)
        beta = self.bias.data.reshape(1, C, 1, 1)
        out_data = gamma * x_norm + beta

        out = Tensor(out_data, (x, self.weight, self.bias), x.requires_grad)

        def _backward() -> None:
            dout = out.grad
            
            # 權重梯度
            self.weight.grad += np.sum(dout * x_norm, axis=(0, 2, 3))
            self.bias.grad += np.sum(dout, axis=(0, 2, 3))

            # 修復：正確的 BatchNorm 反向傳播微積分公式
            if x.requires_grad:
                if self.training:
                    m = N * H * W
                    dx_norm = dout * gamma
                    dx = std_inv / m * (
                        m * dx_norm -
                        np.sum(dx_norm, axis=(0, 2, 3), keepdims=True) -
                        x_norm * np.sum(dx_norm * x_norm, axis=(0, 2, 3), keepdims=True)
                    )
                else:
                    # eval 模式的 backprop (罕見但需實作以防萬一)
                    dx = dout * gamma * std_inv
                x.grad += dx

        out._backward = _backward
        return out

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True


class Dropout2d(Module):
    def __init__(self, p: float = 0.5) -> None:
        self.p = p
        self.training = True

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        batch_size, channels, _, _ = x.data.shape
        
        # 修復：Dropout2d 應該捨棄整層 Channel，所以 Mask 尺寸為 (B, C, 1, 1)
        mask_shape = (batch_size, channels, 1, 1)
        mask = np.random.binomial(1, 1 - self.p, mask_shape).astype(np.float32)
        
        out_data = x.data * mask / (1 - self.p)
        out = Tensor(out_data, (x,), x.requires_grad)

        def _backward() -> None:
            x.grad += out.grad * mask / (1 - self.p)

        out._backward = _backward
        return out

    def eval(self) -> None:
        self.training = False

    def train(self) -> None:
        self.training = True