import numpy as np
import pytest

# 直接引入你真實的模組，不再使用 Mock，避免破壞你的程式架構
from nn.cnn import Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d
from nn.tensor import Tensor

def compute_numerical_gradient(forward_fn, params_to_check, dout, eps=1e-3):
    """
    計算數值梯度 (Numerical Gradient)
    使用 eps=1e-3，這是針對 float32 精度最適合的微小變化量。
    若 eps 太小 (如 1e-5)，float32 會發生嚴重的截斷誤差 (Truncation Error)，導致梯度算錯。
    """
    numerical_grads = []
    
    for p in params_to_check:
        if p is None or not p.requires_grad:
            numerical_grads.append(None)
            continue
            
        num_grad = np.zeros_like(p.data)
        it = np.nditer(p.data, flags=['multi_index'], op_flags=['readwrite'])
        
        while not it.finished:
            idx = it.multi_index
            orig_val = p.data[idx]
            
            p.data[idx] = orig_val + eps
            out_plus = forward_fn()
            loss_plus = np.sum(out_plus.data * dout)
            
            p.data[idx] = orig_val - eps
            out_minus = forward_fn()
            loss_minus = np.sum(out_minus.data * dout)
            
            p.data[idx] = orig_val  # 復原
            num_grad[idx] = (loss_plus - loss_minus) / (2 * eps)
            it.iternext()
            
        numerical_grads.append(num_grad)
        
    return numerical_grads

def verify_gradients(layer, x, params_to_check, atol=5e-2, rtol=5e-2):
    """
    因為真實的 Tensor 預設為 float32，
    我們將絕對誤差(atol)與相對誤差(rtol)設定為 0.05。
    這足以抓出所有矩陣形狀、轉置或數學公式寫錯的 Bug，同時放行 float32 的正常進位誤差。
    """
    out = layer(x)
    dout = np.random.randn(*out.data.shape).astype(np.float32)
    
    # 1. 計算解析梯度 (Analytical Gradient - 也就是你 cnn.py 裡寫的 _backward)
    for p in params_to_check:
        if p is not None and p.requires_grad:
            p.grad = np.zeros_like(p.data)
            
    out = layer(x)
    out.grad = dout.copy()
    out._backward()
    
    analytical_grads = []
    for p in params_to_check:
        if p is not None and p.requires_grad:
            analytical_grads.append(p.grad.copy())
        else:
            analytical_grads.append(None)

    # 2. 計算數值梯度 (Numerical Gradient - 用微積分硬算)
    numerical_grads = compute_numerical_gradient(lambda: layer(x), params_to_check, dout, eps=1e-3)
    
    # 3. 嚴格比對兩者
    for i, (ag, ng) in enumerate(zip(analytical_grads, numerical_grads)):
        if ag is not None and ng is not None:
            np.testing.assert_allclose(
                ag, ng, 
                atol=atol, rtol=rtol, 
                err_msg=f"Layer {layer.__class__.__name__} - Gradient mismatch at parameter index {i}"
            )

# ==================== 測試案例 ====================

def test_conv2d_gradient():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 2, 5, 5).astype(np.float32), requires_grad=True)
    layer = Conv2d(in_channels=2, out_channels=3, kernel_size=3, padding=1, stride=2)
    # 覆蓋隨機權重避免 ReLU 等死區
    layer.weight.data = np.random.randn(*layer.weight.data.shape).astype(np.float32)
    layer.bias.data = np.random.randn(*layer.bias.data.shape).astype(np.float32)
    verify_gradients(layer, x, [x, layer.weight, layer.bias])

def test_maxpool2d_gradient():
    np.random.seed(42)
    # 用 permutation 確保池化區域內沒有重複的最大值 (避免 Ties 造成不可微的交界點)
    x_data = np.random.permutation(np.arange(32)).astype(np.float32).reshape(2, 1, 4, 4)
    x = Tensor(x_data, requires_grad=True)
    layer = MaxPool2d(kernel_size=2, stride=2)
    verify_gradients(layer, x, [x])

def test_avgpool2d_gradient():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 2, 4, 4).astype(np.float32), requires_grad=True)
    layer = AvgPool2d(kernel_size=2, stride=2)
    verify_gradients(layer, x, [x])

def test_batchnorm2d_gradient():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
    layer = BatchNorm2d(num_channels=3)
    layer.weight.data = np.random.randn(*layer.weight.data.shape).astype(np.float32)
    layer.bias.data = np.random.randn(*layer.bias.data.shape).astype(np.float32)
    layer.train() 
    verify_gradients(layer, x, [x, layer.weight, layer.bias])

def test_flatten_gradient():
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
    layer = Flatten()
    verify_gradients(layer, x, [x])