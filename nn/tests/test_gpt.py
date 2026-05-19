"""tests/test_gpt.py - GPT model tests."""
import numpy as np
from nn import Tensor, GPT

def test_gpt_creation():
    model = GPT(vocab_size=20, block_size=16, n_layer=1, n_embd=16, n_head=4)
    params = model.parameters()
    assert len(params) > 0
    print("  [PASS] GPT creation")

def test_gpt_forward():
    model = GPT(vocab_size=20, block_size=16, n_layer=1, n_embd=16, n_head=4)
    idx = Tensor([[1, 2, 3, 4]], requires_grad=False)
    logits, caches = model(idx, kv_caches=None)
    B, T, V = logits.data.shape
    assert B == 1
    assert T == 4
    assert V == 20
    print("  [PASS] GPT forward pass")

def test_gpt_backward():
    model = GPT(vocab_size=20, block_size=16, n_layer=1, n_embd=16, n_head=4)
    idx = Tensor([[1, 2, 3, 4]], requires_grad=False)
    logits, _ = model(idx, kv_caches=None)
    targets = Tensor([[2, 3, 4, 5]])
    loss = logits.cross_entropy(targets)
    loss.backward()
    has_grad = any(p.grad is not None and np.any(p.grad != 0) for p in model.parameters())
    assert has_grad
    print("  [PASS] GPT backward pass")

def test_gpt_kv_cache():
    model = GPT(vocab_size=20, block_size=16, n_layer=1, n_embd=16, n_head=4)
    x1 = Tensor([[1]], requires_grad=False)
    logits1, caches1 = model(x1, kv_caches=None)
    x2 = Tensor([[2]], requires_grad=False)
    logits2, caches2 = model(x2, kv_caches=caches1)
    assert logits2.data.shape == (1, 1, 20)
    print("  [PASS] GPT KV cache")

def test_gpt_multi_layer():
    model = GPT(vocab_size=20, block_size=16, n_layer=2, n_embd=16, n_head=4)
    idx = Tensor([[1, 2, 3]], requires_grad=False)
    logits, caches = model(idx, kv_caches=None)
    assert logits.data.shape == (1, 3, 20)
    assert len(caches) == 2
    print("  [PASS] GPT multi-layer")

if __name__ == "__main__":
    print("\n=== GPT Tests ===")
    test_gpt_creation()
    test_gpt_forward()
    test_gpt_backward()
    test_gpt_kv_cache()
    test_gpt_multi_layer()
    print("\n  all passed")
