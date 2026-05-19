"""tests/test_nn.py - Module, Linear, Embedding, RMSNorm, Adam tests."""
import numpy as np
from nn import Tensor, Module, Linear, Embedding, RMSNorm, Adam

def test_module_parameters():
    class DummyModule(Module):
        def __init__(self):
            self.p1 = Tensor([1.0], requires_grad=True)
            self.p2 = Tensor([2.0], requires_grad=True)
    
    m = DummyModule()
    params = m.parameters()
    assert len(params) == 2
    print("  [PASS] Module.parameters()")

def test_linear_no_bias():
    linear = Linear(3, 4)
    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    y = linear(x)
    assert y.data.shape == (2, 4)
    params = linear.parameters()
    assert len(params) == 1
    print("  [PASS] Linear (no bias)")

def test_linear_with_bias():
    linear = Linear(3, 4, bias=True)
    x = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
    y = linear(x)
    assert y.data.shape == (2, 4)
    params = linear.parameters()
    assert len(params) == 2
    print("  [PASS] Linear (with bias)")

def test_embedding():
    embed = Embedding(10, 4)
    indices = Tensor([1, 3, 5, 3, 1], requires_grad=False)
    out = embed(indices)
    assert out.data.shape == (5, 4)
    print("  [PASS] Embedding")

def test_embedding_backward():
    embed = Embedding(10, 4)
    indices = Tensor([1, 3, 5], requires_grad=False)
    out = embed(indices)
    loss = out.sum()
    loss.backward()
    assert embed.weight.grad is not None
    print("  [PASS] Embedding backward")

def test_rmsnorm():
    norm = RMSNorm(4)
    x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
    y = norm(x)
    assert y.data.shape == (2, 4)
    print("  [PASS] RMSNorm")

def test_rmsnorm_backward():
    norm = RMSNorm(4)
    x = Tensor([[1.0, 2.0, 3.0, 4.0]], requires_grad=True)
    y = norm(x)
    loss = y.sum()
    loss.backward()
    print("  [PASS] RMSNorm backward")

def test_adam_optimizer():
    p = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    p.grad = np.array([0.1, 0.2, 0.3])
    optim = Adam([p], lr=0.01)
    old_data = p.data.copy()
    optim.step()
    assert not np.allclose(p.data, old_data)
    optim.zero_grad()
    assert np.allclose(p.grad, 0.0)
    print("  [PASS] Adam optimizer")

def test_linear_backward():
    linear = Linear(2, 1)
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    y = linear(x)
    loss = y.sum()
    loss.backward()
    assert linear.weight.grad is not None
    print("  [PASS] Linear backward")

if __name__ == "__main__":
    print("\n=== nn Module Tests ===")
    test_module_parameters()
    test_linear_no_bias()
    test_linear_with_bias()
    test_embedding()
    test_embedding_backward()
    test_rmsnorm()
    test_rmsnorm_backward()
    test_adam_optimizer()
    test_linear_backward()
    print("\n  all passed")
