"""
pytest test suite for the mynn autograd package.
Covers forward values, backward (gradient) correctness via numerical
finite-difference comparison, and edge cases.
"""

import sys
import pytest
import numpy as np

from nn.tensor import Tensor, cat
from nn.nn import (
    mse_loss, Linear, Embedding, RMSNorm, Adam,
    Sequential, ReLU, Tanh, Module,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ATOL = 1e-5
RTOL = 5e-2   # float32 numerical-grad tolerance
EPS  = 1e-3   # finite-difference step (float32 friendly)


def numerical_grad(f, param: Tensor) -> np.ndarray:
    """Compute numerical gradient of scalar f() w.r.t. param via central diff."""
    grad = np.zeros_like(param.data)
    it = np.nditer(param.data, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        orig = float(param.data[idx])
        param.data[idx] = np.float32(orig + EPS)
        fp = float(f().data.sum())
        param.data[idx] = np.float32(orig - EPS)
        fm = float(f().data.sum())
        param.data[idx] = np.float32(orig)
        grad[idx] = (fp - fm) / (2 * EPS)
        it.iternext()
    return grad


def grad_check(make_fn, tol=RTOL):
    """
    Run a gradient check.
    make_fn() must return ([(name, Tensor), ...], callable -> Tensor).
    Asserts that analytical and numerical gradients agree within tol.
    """
    params, f = make_fn()
    for name, p in params:
        # Analytical
        for _, pp in params:
            pp.zero_grad()
        out = f()
        out.backward()
        analytic = p.grad.copy()
        # Numerical
        for _, pp in params:
            pp.zero_grad()
        num = numerical_grad(f, p)
        rel_err = np.abs(num - analytic) / (np.abs(num) + np.abs(analytic) + 1e-6)
        assert rel_err.max() < tol, (
            f"Gradient check FAILED for '{name}': "
            f"max_rel_err={rel_err.max():.4e} (tol={tol})\n"
            f"  numeric  = {num.flatten()[:6]}\n"
            f"  analytic = {analytic.flatten()[:6]}"
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)


# ===========================================================================
# 1. Basic arithmetic ops
# ===========================================================================

class TestArithmetic:

    def test_add_forward(self):
        a = Tensor([1.0, 2.0, 3.0])
        b = Tensor([4.0, 5.0, 6.0])
        out = a + b
        np.testing.assert_allclose(out.data, [5, 7, 9], atol=ATOL)

    def test_add_grad(self):
        def make():
            a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
            b = Tensor([4.0, 5.0, 6.0], requires_grad=True)
            return [("a", a), ("b", b)], lambda: a + b
        grad_check(make)

    def test_add_scalar(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        out = x + 3.0
        np.testing.assert_allclose(out.data, [4.0, 5.0], atol=ATOL)

    def test_mul_forward(self):
        a = Tensor([2.0, 3.0])
        b = Tensor([4.0, -1.0])
        np.testing.assert_allclose((a * b).data, [8.0, -3.0], atol=ATOL)

    def test_mul_grad(self):
        def make():
            a = Tensor([2.0, -1.0, 0.5], requires_grad=True)
            b = Tensor([3.0,  4.0, -2.0], requires_grad=True)
            return [("a", a), ("b", b)], lambda: a * b
        grad_check(make)

    def test_sub_forward(self):
        a = Tensor([5.0, 3.0])
        b = Tensor([2.0, 4.0])
        np.testing.assert_allclose((a - b).data, [3.0, -1.0], atol=ATOL)

    def test_neg(self):
        x = Tensor([1.0, -2.0], requires_grad=True)
        out = -x
        np.testing.assert_allclose(out.data, [-1.0, 2.0], atol=ATOL)

    def test_pow_forward(self):
        x = Tensor([2.0, 3.0])
        np.testing.assert_allclose((x ** 2).data, [4.0, 9.0], atol=ATOL)

    def test_pow_grad(self):
        def make():
            x = Tensor([1.0, 2.0, -1.0], requires_grad=True)
            return [("x", x)], lambda: x ** 3
        grad_check(make)

    def test_matmul_forward(self):
        A = Tensor([[1.0, 2.0], [3.0, 4.0]])
        B = Tensor([[1.0, 0.0], [0.0, 1.0]])
        np.testing.assert_allclose((A @ B).data, A.data, atol=ATOL)

    def test_matmul_grad(self):
        def make():
            A = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            B = Tensor(np.random.randn(4, 2).astype(np.float32), requires_grad=True)
            return [("A", A), ("B", B)], lambda: A @ B
        grad_check(make)

    def test_broadcast_add_grad(self):
        def make():
            a = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            b = Tensor(np.random.randn(4).astype(np.float32), requires_grad=True)
            return [("a", a), ("b", b)], lambda: a + b
        grad_check(make)


# ===========================================================================
# 2. Reduction ops  (the fixed bugs live here)
# ===========================================================================

class TestReductions:

    def test_sum_global_forward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        assert float(x.sum().data) == pytest.approx(10.0)

    def test_sum_global_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.sum()
        grad_check(make)

    def test_sum_axis0_forward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(x.sum(axis=0).data, [4.0, 6.0], atol=ATOL)

    def test_sum_axis0_grad(self):
        """BUG 1 regression: sum(axis=0) used to crash with shape mismatch."""
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.sum(axis=0)
        grad_check(make)

    def test_sum_axis1_forward(self):
        x = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        np.testing.assert_allclose(x.sum(axis=1).data, [6.0, 15.0], atol=ATOL)

    def test_sum_axis1_grad(self):
        """BUG 1 regression: sum(axis=1) used to crash with shape mismatch."""
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.sum(axis=1)
        grad_check(make)

    def test_sum_keepdims(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        out = x.sum(axis=1, keepdims=True)
        assert out.shape == (2, 1)
        np.testing.assert_allclose(out.data, [[3.0], [7.0]], atol=ATOL)

    def test_sum_keepdims_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.sum(axis=1, keepdims=True)
        grad_check(make)

    def test_mean_global_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.mean()
        grad_check(make)

    def test_mean_axis1_grad(self):
        """BUG 1 regression: mean(axis=1) used to crash like sum(axis=1)."""
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.mean(axis=1)
        grad_check(make)

    def test_mean_axis_value(self):
        x = Tensor([[2.0, 4.0], [6.0, 8.0]])
        np.testing.assert_allclose(x.mean(axis=1).data, [3.0, 7.0], atol=ATOL)


# ===========================================================================
# 3. Activation functions
# ===========================================================================

class TestActivations:

    def test_relu_forward(self):
        x = Tensor([-1.0, 0.0, 2.0])
        np.testing.assert_allclose(x.relu().data, [0.0, 0.0, 2.0], atol=ATOL)

    def test_relu_grad(self):
        def make():
            x = Tensor([-1.0, 0.5, 2.0, -0.1], requires_grad=True)
            return [("x", x)], lambda: x.relu()
        grad_check(make)

    def test_tanh_forward(self):
        x = Tensor([0.0])
        assert float(x.tanh().data[0]) == pytest.approx(0.0, abs=ATOL)

    def test_tanh_grad(self):
        def make():
            x = Tensor([-1.0, 0.0, 1.0, 2.0], requires_grad=True)
            return [("x", x)], lambda: x.tanh()
        grad_check(make)

    def test_softmax_sums_to_one(self):
        x = Tensor(np.random.randn(4, 6).astype(np.float32))
        s = x.softmax()
        np.testing.assert_allclose(s.data.sum(axis=-1), np.ones(4), atol=1e-5)

    def test_softmax_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            w = Tensor(np.random.randn(3, 4).astype(np.float32))
            return [("x", x)], lambda: (x.softmax() * w).sum()
        grad_check(make)


# ===========================================================================
# 4. Shape / indexing ops
# ===========================================================================

class TestShapeOps:

    def test_reshape_forward(self):
        x = Tensor(np.arange(12).reshape(3, 4).astype(np.float32))
        assert x.reshape(2, 6).shape == (2, 6)

    def test_reshape_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.reshape(2, 6)
        grad_check(make)

    def test_transpose_forward(self):
        x = Tensor([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_allclose(x.transpose(0, 1).data, [[1, 3], [2, 4]], atol=ATOL)

    def test_transpose_grad(self):
        def make():
            x = Tensor(np.random.randn(3, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.transpose(0, 1)
        grad_check(make)

    def test_cat_forward(self):
        a = Tensor([[1.0, 2.0], [3.0, 4.0]])
        b = Tensor([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0]])
        out = cat([a, b], axis=1)
        assert out.shape == (2, 5)
        np.testing.assert_allclose(out.data[0], [1, 2, 5, 6, 7], atol=ATOL)

    def test_cat_grad(self):
        def make():
            a = Tensor(np.random.randn(2, 3).astype(np.float32), requires_grad=True)
            b = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
            return [("a", a), ("b", b)], lambda: cat([a, b], axis=1)
        grad_check(make)


# ===========================================================================
# 5. Masking / clamping
# ===========================================================================

class TestMaskClamp:

    def test_masked_fill_forward(self):
        x = Tensor([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        out = x.masked_fill(mask, -999.0)
        np.testing.assert_allclose(out.data, [-999, 2, -999, 4], atol=ATOL)

    def test_masked_fill_grad(self):
        mask = np.array([[True, False], [False, True]])
        def make():
            x = Tensor(np.random.randn(2, 2).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: x.masked_fill(mask, 0.0)
        grad_check(make)

    def test_masked_fill_blocks_grad_at_true(self):
        """Gradient must be zero where mask is True."""
        x = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
        mask = np.array([True, False, True, False])
        x.masked_fill(mask, 0.0).sum().backward()
        np.testing.assert_allclose(x.grad[[0, 2]], [0.0, 0.0], atol=ATOL)
        np.testing.assert_allclose(x.grad[[1, 3]], [1.0, 1.0], atol=ATOL)

    def test_clamp_both_forward(self):
        x = Tensor([-3.0, 0.5, 3.0])
        np.testing.assert_allclose(x.clamp(-1.0, 1.0).data, [-1, 0.5, 1], atol=ATOL)

    def test_clamp_both_grad(self):
        def make():
            x = Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
            return [("x", x)], lambda: x.clamp(-1.0, 1.0)
        grad_check(make)

    def test_clamp_min_only_forward(self):
        x = Tensor([-2.0, 0.5, 3.0])
        np.testing.assert_allclose(x.clamp(min_val=0.0).data, [0, 0.5, 3], atol=ATOL)

    def test_clamp_min_only_grad(self):
        """BUG 3 regression: clamp(min=x) used to let gradient through clamped positions."""
        def make():
            x = Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
            return [("x", x)], lambda: x.clamp(min_val=0.0)
        grad_check(make)

    def test_clamp_min_only_grad_value(self):
        """Directly verify: clamped position gets zero gradient."""
        x = Tensor([-2.0, 1.0, 3.0], requires_grad=True)
        x.clamp(min_val=0.0).sum().backward()
        assert x.grad[0] == pytest.approx(0.0, abs=ATOL), "clamped element should have grad=0"
        assert x.grad[1] == pytest.approx(1.0, abs=ATOL)
        assert x.grad[2] == pytest.approx(1.0, abs=ATOL)

    def test_clamp_max_only_grad(self):
        """BUG 3 regression: same for max-only clamp."""
        def make():
            x = Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
            return [("x", x)], lambda: x.clamp(max_val=0.0)
        grad_check(make)

    def test_abs_forward(self):
        x = Tensor([-3.0, 0.0, 2.0])
        np.testing.assert_allclose(x.abs().data, [3, 0, 2], atol=ATOL)

    def test_abs_grad(self):
        def make():
            x = Tensor([-2.0, -0.5, 0.5, 2.0], requires_grad=True)
            return [("x", x)], lambda: x.abs()
        grad_check(make)


# ===========================================================================
# 6. Loss functions
# ===========================================================================

class TestLossFunctions:

    # --- mse_loss ---

    def test_mse_loss_forward(self):
        pred   = Tensor([1.0, 2.0, 3.0])
        target = Tensor([1.0, 2.0, 3.0])
        assert float(mse_loss(pred, target).data) == pytest.approx(0.0, abs=ATOL)

    def test_mse_loss_forward_value(self):
        pred   = Tensor([0.0, 0.0])
        target = Tensor([1.0, 3.0])
        # mse = mean([1, 9]) = 5
        assert float(mse_loss(pred, target).data) == pytest.approx(5.0, abs=1e-4)

    def test_mse_loss_grad(self):
        """BUG 2 regression: mse_loss used to produce zero gradients."""
        def make():
            pred   = Tensor([1.0, 2.0, 3.0], requires_grad=True)
            target = Tensor([1.5, 2.5, 3.5], requires_grad=False)
            return [("pred", pred)], lambda: mse_loss(pred, target)
        grad_check(make)

    def test_mse_loss_grad_not_zero(self):
        """Direct regression: gradient must be non-zero when pred != target."""
        pred   = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([1.5, 2.5, 3.5], requires_grad=False)
        mse_loss(pred, target).backward()
        assert np.any(pred.grad != 0.0), "mse_loss gradient must not be all-zero"

    def test_mse_loss_grad_value(self):
        """grad = 2*(pred-target)/N."""
        pred   = Tensor([1.0, 2.0, 3.0], requires_grad=True)
        target = Tensor([1.5, 2.5, 3.5], requires_grad=False)
        mse_loss(pred, target).backward()
        expected = 2 * (pred.data - target.data) / 3
        np.testing.assert_allclose(pred.grad, expected, atol=1e-5)

    # --- cross_entropy ---

    def test_cross_entropy_forward_shape(self):
        logits = Tensor(np.random.randn(2, 3, 5).astype(np.float32), requires_grad=True)
        targets = np.array([[1, 2, 0], [3, 4, 1]])
        loss = logits.cross_entropy(targets)
        assert loss.data.shape == ()   # scalar

    def test_cross_entropy_forward_nonneg(self):
        logits = Tensor(np.random.randn(2, 3, 5).astype(np.float32))
        targets = np.array([[1, 2, 0], [3, 4, 1]])
        assert float(logits.cross_entropy(targets).data) > 0

    def test_cross_entropy_perfect_prediction(self):
        """Perfect logits → loss close to 0."""
        logits = np.zeros((1, 1, 3), dtype=np.float32)
        logits[0, 0, 2] = 100.0          # class 2 is overwhelmingly dominant
        loss = Tensor(logits).cross_entropy(np.array([[2]]))
        assert float(loss.data) < 1e-3

    def test_cross_entropy_grad(self):
        targets = np.array([[1, 2, 0], [3, 4, 1]])
        def make():
            logits = Tensor(np.random.randn(2, 3, 5).astype(np.float32), requires_grad=True)
            return [("logits", logits)], lambda: logits.cross_entropy(targets)
        grad_check(make)


# ===========================================================================
# 7. Layers
# ===========================================================================

class TestLayers:

    def test_linear_forward_shape(self):
        layer = Linear(4, 8)
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        assert layer(x).shape == (3, 8)

    def test_linear_bias_forward(self):
        layer = Linear(2, 3, bias=True)
        layer.bias.data[:] = 1.0
        layer.weight.data[:] = 0.0
        x = Tensor(np.ones((1, 2), dtype=np.float32))
        np.testing.assert_allclose(layer(x).data, [[1, 1, 1]], atol=ATOL)

    def test_linear_grad(self):
        def make():
            layer = Linear(4, 3, bias=True)
            x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
            return [("x", x), ("W", layer.weight), ("b", layer.bias)], lambda: layer(x)
        grad_check(make)

    def test_embedding_forward_shape(self):
        emb = Embedding(10, 4)
        out = emb(np.array([1, 3, 5]))
        assert out.shape == (3, 4)

    def test_embedding_forward_values(self):
        emb = Embedding(10, 4)
        idx = np.array([2, 5])
        out = emb(idx)
        np.testing.assert_allclose(out.data[0], emb.weight.data[2], atol=ATOL)
        np.testing.assert_allclose(out.data[1], emb.weight.data[5], atol=ATOL)

    def test_embedding_grad(self):
        emb_w = np.random.randn(10, 4).astype(np.float32)
        def make():
            emb = Embedding(10, 4)
            emb.weight.data = emb_w.copy()
            idx = np.array([1, 3, 1, 5])
            return [("weight", emb.weight)], lambda: emb(idx)
        grad_check(make)

    def test_embedding_duplicate_indices_grad(self):
        """np.add.at must accumulate grads for repeated indices."""
        emb = Embedding(5, 3)
        idx = np.array([1, 1])   # index 1 appears twice
        emb.weight.zero_grad()
        emb(idx).sum().backward()
        # grad at row 1 should be 2× (two lookups)
        np.testing.assert_allclose(emb.weight.grad[1], np.ones(3) * 2, atol=ATOL)

    def test_rmsnorm_output_shape(self):
        norm = RMSNorm(8)
        x = Tensor(np.random.randn(4, 8).astype(np.float32), requires_grad=True)
        assert norm(x).shape == (4, 8)

    def test_rmsnorm_rms_equals_one(self):
        """After RMSNorm, RMS of each row should be ~1."""
        norm = RMSNorm(16, eps=0)
        x = Tensor(np.random.randn(4, 16).astype(np.float32))
        out = norm(x)
        rms = np.sqrt(np.mean(out.data ** 2, axis=-1))
        np.testing.assert_allclose(rms, np.ones(4), atol=1e-4)

    def test_rmsnorm_grad(self):
        def make():
            x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
            norm = RMSNorm(4)
            return [("x", x)], lambda: norm(x)
        grad_check(make)

    def test_sequential_forward(self):
        model = Sequential(Linear(4, 8), ReLU(), Linear(8, 2))
        x = Tensor(np.random.randn(3, 4).astype(np.float32))
        assert model(x).shape == (3, 2)

    def test_relu_module(self):
        x = Tensor([-1.0, 2.0])
        np.testing.assert_allclose(ReLU()(x).data, [0, 2], atol=ATOL)

    def test_tanh_module(self):
        x = Tensor([0.0])
        assert float(Tanh()(x).data[0]) == pytest.approx(0.0, abs=ATOL)


# ===========================================================================
# 8. Module.parameters()
# ===========================================================================

class TestModuleParameters:

    def test_linear_parameters(self):
        layer = Linear(4, 8)
        assert layer.weight in layer.parameters()

    def test_linear_bias_parameters(self):
        layer = Linear(4, 8, bias=True)
        params = layer.parameters()
        assert layer.weight in params
        assert layer.bias in params

    def test_sequential_parameters(self):
        model = Sequential(Linear(4, 8, bias=True), Linear(8, 2, bias=True))
        params = model.parameters()
        assert len(params) == 4   # 2 weights + 2 biases

    def test_no_requires_grad_excluded(self):
        """Tensors with requires_grad=False must not appear in parameters()."""
        norm = RMSNorm(4)   # scale has requires_grad=False
        assert norm.scale not in norm.parameters()


# ===========================================================================
# 9. Adam optimizer
# ===========================================================================

class TestAdam:

    def test_step_decreases_loss(self):
        """Adam should reduce MSE loss in a few steps."""
        np.random.seed(42)
        pred   = Tensor(np.array([0.0, 0.0, 0.0]), requires_grad=True)
        target = Tensor(np.array([1.0, 2.0, 3.0]))
        opt = Adam([pred], lr=0.1)

        loss_before = float(mse_loss(pred, target).data)
        for _ in range(20):
            opt.zero_grad()
            loss = mse_loss(pred, target)
            loss.backward()
            opt.step()
        loss_after = float(mse_loss(pred, target).data)

        assert loss_after < loss_before, "Adam did not reduce loss"

    def test_zero_grad_clears_grads(self):
        p = Tensor(np.array([1.0, 2.0]), requires_grad=True)
        opt = Adam([p], lr=0.01)
        (p * Tensor([1.0, 1.0])).sum().backward()
        assert np.any(p.grad != 0)
        opt.zero_grad()
        np.testing.assert_allclose(p.grad, [0.0, 0.0], atol=ATOL)

    def test_adam_converges_linear_regression(self):
        """Train a single Linear layer to fit y = 2x."""
        np.random.seed(7)
        layer = Linear(1, 1, bias=True)
        opt = Adam(layer.parameters(), lr=0.05)

        x_np = np.linspace(-1, 1, 20).reshape(-1, 1).astype(np.float32)
        y_np = (2 * x_np).astype(np.float32)

        for _ in range(200):
            opt.zero_grad()
            x = Tensor(x_np)
            y = Tensor(y_np)
            loss = mse_loss(layer(x), y)
            loss.backward()
            opt.step()

        assert float(loss.data) < 0.01, f"Loss too high: {float(loss.data)}"


# ===========================================================================
# 10. Backward graph integrity
# ===========================================================================

class TestBackwardGraph:

    def test_zero_grad_resets(self):
        x = Tensor([1.0, 2.0], requires_grad=True)
        (x * x).sum().backward()
        assert np.any(x.grad != 0)
        x.zero_grad()
        np.testing.assert_allclose(x.grad, [0.0, 0.0], atol=ATOL)

    def test_grad_accumulates(self):
        """Calling backward twice accumulates gradients."""
        x = Tensor([3.0], requires_grad=True)
        (x * x).sum().backward()
        g1 = x.grad.copy()
        (x * x).sum().backward()   # accumulate
        np.testing.assert_allclose(x.grad, 2 * g1, atol=ATOL)

    def test_chain_rule(self):
        """(x^2 + 1) -> tanh -> sum: check d/dx numerically."""
        def make():
            x = Tensor(np.array([-0.5, 0.5, 1.0]), requires_grad=True)
            return [("x", x)], lambda: (x ** 2 + Tensor(1.0)).tanh()
        grad_check(make)

    def test_deep_chain(self):
        """Several ops chained: relu(matmul(x, W) + b)."""
        def make():
            x = Tensor(np.random.randn(2, 4).astype(np.float32), requires_grad=True)
            W = Tensor(np.random.randn(4, 3).astype(np.float32), requires_grad=True)
            b = Tensor(np.random.randn(3).astype(np.float32), requires_grad=True)
            return [("x", x), ("W", W), ("b", b)], lambda: (x @ W + b).relu()
        grad_check(make)