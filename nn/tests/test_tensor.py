"""tests/test_tensor.py - Tensor autodiff tests."""
import numpy as np
from nn import Tensor, cat

def test_tensor_creation():
    t = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert t.data.shape == (2, 2)
    assert t.requires_grad == True
    print("  [PASS] Tensor creation")

def test_tensor_add():
    a = Tensor([1.0, 2.0], requires_grad=True)
    b = Tensor([3.0, 4.0], requires_grad=True)
    c = a + b
    assert np.allclose(c.data, [4.0, 6.0])
    print("  [PASS] Tensor add")

def test_tensor_mul():
    a = Tensor([2.0, 3.0], requires_grad=True)
    b = Tensor([4.0, 5.0], requires_grad=True)
    c = a * b
    assert np.allclose(c.data, [8.0, 15.0])
    print("  [PASS] Tensor mul")

def test_tensor_matmul():
    a = Tensor([[1, 2], [3, 4]], requires_grad=True)
    b = Tensor([[5, 6], [7, 8]], requires_grad=True)
    c = a @ b
    assert np.allclose(c.data, [[19, 22], [43, 50]])
    print("  [PASS] Tensor matmul")

def test_tensor_backward():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([0.5, 1.0, 1.5], requires_grad=True)
    c = a * b
    loss = c.sum()
    loss.backward()
    assert np.allclose(a.grad, [0.5, 1.0, 1.5])
    assert np.allclose(b.grad, [1.0, 2.0, 3.0])
    print("  [PASS] Tensor backward")

def test_tensor_relu():
    x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
    y = x.relu()
    assert np.allclose(y.data, [0.0, 0.0, 1.0, 2.0])
    print("  [PASS] Tensor relu")

def test_tensor_softmax():
    x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)
    y = x.softmax(axis=-1)
    assert np.allclose(y.data.sum(axis=-1), 1.0)
    print("  [PASS] Tensor softmax")

def test_tensor_transpose():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    y = x.transpose(0, 1)
    assert y.data.shape == (3, 2)
    print("  [PASS] Tensor transpose")

def test_tensor_cat():
    a = Tensor([1, 2, 3], requires_grad=True)
    b = Tensor([4, 5], requires_grad=True)
    c = cat([a, b], axis=0)
    assert np.allclose(c.data, [1, 2, 3, 4, 5])
    print("  [PASS] cat")

if __name__ == "__main__":
    print("\n=== Tensor Tests ===")
    test_tensor_creation()
    test_tensor_add()
    test_tensor_mul()
    test_tensor_matmul()
    test_tensor_backward()
    test_tensor_relu()
    test_tensor_softmax()
    test_tensor_transpose()
    test_tensor_cat()
    print("\n  all passed")
