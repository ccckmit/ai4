"""tests/test_cnn.py - CNN layer tests with type annotation and gradient checking."""

from __future__ import annotations

import numpy as np
import pytest
from nn import Tensor, Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d, ReLU


ATOL = 1e-5
RTOL = 5e-2
EPS = 1e-3


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
    """Run a gradient check."""
    params, f = make_fn()
    for name, p in params:
        for _, pp in params:
            pp.zero_grad()
        out = f()
        out.backward()
        analytic = p.grad.copy()
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


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(0)


class TestConv2d:
    """Tests for Conv2d layer."""

    def test_conv2d_creation(self) -> None:
        """Test Conv2d layer creation."""
        conv = Conv2d(in_channels=3, out_channels=16, kernel_size=3)
        assert conv.in_channels == 3
        assert conv.out_channels == 16
        assert conv.kernel_size == 3
        assert conv.weight.data.shape == (16, 3, 3, 3)
        print("  [PASS] Conv2d creation")

    def test_conv2d_forward(self) -> None:
        """Test Conv2d forward pass."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=0)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        assert out.data.shape == (2, 8, 8, 8)
        print("  [PASS] Conv2d forward")

    def test_conv2d_with_padding(self) -> None:
        """Test Conv2d with padding."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        assert out.data.shape == (2, 8, 10, 10)
        print("  [PASS] Conv2d with padding")

    def test_conv2d_with_stride(self) -> None:
        """Test Conv2d with stride."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        expected_h = (10 - 3) // 2 + 1
        expected_w = (10 - 3) // 2 + 1
        assert out.data.shape == (2, 8, expected_h, expected_w)
        print("  [PASS] Conv2d with stride")

    def test_conv2d_with_bias(self) -> None:
        """Test Conv2d with bias."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=True)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        assert out.data.shape == (2, 8, 8, 8)
        assert conv.bias is not None
        print("  [PASS] Conv2d with bias")

    def test_conv2d_no_bias(self) -> None:
        """Test Conv2d without bias."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, bias=False)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        assert conv.bias is None
        print("  [PASS] Conv2d no bias")

    def test_conv2d_backward(self) -> None:
        """Test Conv2d backward pass."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.data.shape
        print("  [PASS] Conv2d backward")

    def test_conv2d_weight_grad(self) -> None:
        """Test Conv2d weight gradient computation."""
        np.random.seed(42)
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3, bias=False)
        x = Tensor(np.ones((1, 1, 5, 5)), requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert conv.weight.grad is not None
        assert conv.weight.grad.shape == conv.weight.data.shape
        print("  [PASS] Conv2d weight gradient")

    def test_conv2d_single_channel(self) -> None:
        """Test Conv2d with single input channel."""
        conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3)
        x = Tensor(np.random.randn(1, 1, 10, 10), requires_grad=True)
        out = conv(x)
        assert out.data.shape == (1, 1, 8, 8)
        print("  [PASS] Conv2d single channel")

    def test_conv2d_gradient_check(self) -> None:
        """Gradient check for Conv2d with stride=1 and padding=0."""
        def make():
            conv = Conv2d(in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True)
            x = Tensor(np.random.randn(1, 1, 5, 5).astype(np.float32), requires_grad=True)
            return [("x", x), ("weight", conv.weight), ("bias", conv.bias)], lambda: conv(x)
        grad_check(make, tol=0.15)


class TestMaxPool2d:
    """Tests for MaxPool2d layer."""

    def test_maxpool2d_creation(self) -> None:
        """Test MaxPool2d layer creation."""
        pool = MaxPool2d(kernel_size=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2
        print("  [PASS] MaxPool2d creation")

    def test_maxpool2d_custom_stride(self) -> None:
        """Test MaxPool2d with custom stride."""
        pool = MaxPool2d(kernel_size=2, stride=1)
        assert pool.stride == 1
        print("  [PASS] MaxPool2d custom stride")

    def test_maxpool2d_forward(self) -> None:
        """Test MaxPool2d forward pass."""
        pool = MaxPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = pool(x)
        assert out.data.shape == (2, 3, 5, 5)
        print("  [PASS] MaxPool2d forward")

    def test_maxpool2d_odd_input(self) -> None:
        """Test MaxPool2d with odd input dimensions."""
        pool = MaxPool2d(kernel_size=3, stride=2)
        x = Tensor(np.random.randn(1, 1, 7, 7), requires_grad=True)
        out = pool(x)
        expected = (7 - 3) // 2 + 1
        assert out.data.shape == (1, 1, expected, expected)
        print("  [PASS] MaxPool2d odd input")

    def test_maxpool2d_backward(self) -> None:
        """Test MaxPool2d backward pass."""
        pool = MaxPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.data.shape
        print("  [PASS] MaxPool2d backward")

    def test_maxpool2d_gradient_values(self) -> None:
        """Test MaxPool2d gradient flows to max positions."""
        x_data = np.array([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        pool = MaxPool2d(kernel_size=2)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        expected_grad = np.array([[[[0.0, 0.0], [0.0, 1.0]]]], dtype=np.float32)
        assert np.allclose(x.grad, expected_grad)
        print("  [PASS] MaxPool2d gradient values")


class TestAvgPool2d:
    """Tests for AvgPool2d layer."""

    def test_avgpool2d_creation(self) -> None:
        """Test AvgPool2d layer creation."""
        pool = AvgPool2d(kernel_size=2)
        assert pool.kernel_size == 2
        assert pool.stride == 2
        print("  [PASS] AvgPool2d creation")

    def test_avgpool2d_custom_stride(self) -> None:
        """Test AvgPool2d with custom stride."""
        pool = AvgPool2d(kernel_size=2, stride=1)
        assert pool.stride == 1
        print("  [PASS] AvgPool2d custom stride")

    def test_avgpool2d_forward(self) -> None:
        """Test AvgPool2d forward pass."""
        pool = AvgPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = pool(x)
        assert out.data.shape == (2, 3, 5, 5)
        print("  [PASS] AvgPool2d forward")

    def test_avgpool2d_backward(self) -> None:
        """Test AvgPool2d backward pass."""
        pool = AvgPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.data.shape
        print("  [PASS] AvgPool2d backward")

    def test_avgpool2d_gradient_values(self) -> None:
        """Test AvgPool2d gradient is uniform."""
        x_data = np.ones((1, 1, 4, 4), dtype=np.float32)
        x = Tensor(x_data, requires_grad=True)
        pool = AvgPool2d(kernel_size=2)
        out = pool(x)
        loss = out.sum()
        loss.backward()
        expected_grad = np.ones((1, 1, 4, 4), dtype=np.float32) / 4
        assert np.allclose(x.grad, expected_grad)
        print("  [PASS] AvgPool2d gradient values")


class TestFlatten:
    """Tests for Flatten layer."""

    def test_flatten_creation(self) -> None:
        """Test Flatten layer creation."""
        flat = Flatten()
        print("  [PASS] Flatten creation")

    def test_flatten_forward(self) -> None:
        """Test Flatten forward pass."""
        flat = Flatten()
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = flat(x)
        assert out.data.shape == (2, 300)
        print("  [PASS] Flatten forward")

    def test_flatten_single_batch(self) -> None:
        """Test Flatten with batch size 1."""
        flat = Flatten()
        x = Tensor(np.random.randn(1, 3, 10, 10), requires_grad=True)
        out = flat(x)
        assert out.data.shape == (1, 300)
        print("  [PASS] Flatten single batch")

    def test_flatten_backward(self) -> None:
        """Test Flatten backward pass."""
        flat = Flatten()
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = flat(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == x.data.shape
        print("  [PASS] Flatten backward")

    def test_flatten_gradient_check(self) -> None:
        """Gradient check for Flatten."""
        def make():
            flat = Flatten()
            x = Tensor(np.random.randn(2, 3, 4, 4).astype(np.float32), requires_grad=True)
            return [("x", x)], lambda: flat(x)
        grad_check(make, tol=0.1)


class TestBatchNorm2d:
    """Tests for BatchNorm2d layer."""

    def test_batchnorm2d_creation(self) -> None:
        """Test BatchNorm2d layer creation."""
        bn = BatchNorm2d(num_channels=16)
        assert bn.num_channels == 16
        assert bn.weight.data.shape == (16,)
        assert bn.bias.data.shape == (16,)
        print("  [PASS] BatchNorm2d creation")

    def test_batchnorm2d_forward(self) -> None:
        """Test BatchNorm2d forward pass."""
        bn = BatchNorm2d(num_channels=8)
        x = Tensor(np.random.randn(4, 8, 10, 10), requires_grad=True)
        out = bn(x)
        assert out.data.shape == (4, 8, 10, 10)
        print("  [PASS] BatchNorm2d forward")

    def test_batchnorm2d_eval_mode(self) -> None:
        """Test BatchNorm2d eval mode."""
        bn = BatchNorm2d(num_channels=8)
        bn.eval()
        assert bn.training is False
        print("  [PASS] BatchNorm2d eval mode")

    def test_batchnorm2d_train_mode(self) -> None:
        """Test BatchNorm2d train mode."""
        bn = BatchNorm2d(num_channels=8)
        bn.eval()
        bn.train()
        assert bn.training is True
        print("  [PASS] BatchNorm2d train mode")

    def test_batchnorm2d_backward(self) -> None:
        """Test BatchNorm2d backward pass."""
        bn = BatchNorm2d(num_channels=8)
        x = Tensor(np.random.randn(4, 8, 10, 10), requires_grad=True)
        out = bn(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert bn.weight.grad is not None
        assert bn.bias.grad is not None
        print("  [PASS] BatchNorm2d backward")


class TestDropout2d:
    """Tests for Dropout2d layer."""

    def test_dropout2d_creation(self) -> None:
        """Test Dropout2d layer creation."""
        drop = Dropout2d(p=0.5)
        assert drop.p == 0.5
        print("  [PASS] Dropout2d creation")

    def test_dropout2d_train_mode(self) -> None:
        """Test Dropout2d train mode."""
        drop = Dropout2d(p=0.5)
        assert drop.training is True
        print("  [PASS] Dropout2d train mode")

    def test_dropout2d_eval_mode(self) -> None:
        """Test Dropout2d eval mode passes through."""
        drop = Dropout2d(p=0.5)
        drop.eval()
        assert drop.training is False
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = drop(x)
        assert np.allclose(out.data, x.data)
        print("  [PASS] Dropout2d eval mode")

    def test_dropout2d_forward(self) -> None:
        """Test Dropout2d forward pass in training mode."""
        np.random.seed(42)
        drop = Dropout2d(p=0.5)
        x = Tensor(np.ones((2, 3, 10, 10)), requires_grad=True)
        out = drop(x)
        assert out.data.shape == (2, 3, 10, 10)
        print("  [PASS] Dropout2d forward")


class TestCNNIntegration:
    """Integration tests for CNN layers."""

    def test_conv_relu_pool(self) -> None:
        """Test Conv2d + ReLU + MaxPool2d."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        pool = MaxPool2d(kernel_size=2)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        out = out.relu()
        out = pool(out)
        assert out.data.shape == (2, 8, 5, 5)
        print("  [PASS] Conv + ReLU + Pool")

    def test_conv_batchnorm_relu(self) -> None:
        """Test Conv2d + BatchNorm2d + ReLU."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        bn = BatchNorm2d(num_channels=8)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        out = bn(out)
        out = out.relu()
        assert out.data.shape == (2, 8, 10, 10)
        print("  [PASS] Conv + BatchNorm + ReLU")

    def test_simple_cnn(self) -> None:
        """Test simple CNN architecture."""
        conv1 = Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)
        pool1 = MaxPool2d(kernel_size=2)
        conv2 = Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1)
        pool2 = MaxPool2d(kernel_size=2)
        flat = Flatten()

        x = Tensor(np.random.randn(4, 1, 28, 28), requires_grad=True)

        out = conv1(x)
        out = out.relu()
        out = pool1(out)
        assert out.data.shape == (4, 8, 14, 14)

        out = conv2(out)
        out = out.relu()
        out = pool2(out)
        assert out.data.shape == (4, 16, 7, 7)

        out = flat(out)
        assert out.data.shape == (4, 784)

        print("  [PASS] Simple CNN")

    def test_end_to_end_backward(self) -> None:
        """Test end-to-end backward pass."""
        conv = Conv2d(in_channels=3, out_channels=8, kernel_size=3)
        x = Tensor(np.random.randn(2, 3, 10, 10), requires_grad=True)
        out = conv(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert conv.weight.grad is not None
        print("  [PASS] End-to-end backward")


if __name__ == "__main__":
    print("\n=== CNN Tests ===")
    test_conv = TestConv2d()
    test_conv.test_conv2d_creation()
    test_conv.test_conv2d_forward()
    test_conv.test_conv2d_with_padding()
    test_conv.test_conv2d_with_stride()
    test_conv.test_conv2d_with_bias()
    test_conv.test_conv2d_no_bias()
    test_conv.test_conv2d_backward()
    test_conv.test_conv2d_weight_grad()
    test_conv.test_conv2d_single_channel()

    test_maxpool = TestMaxPool2d()
    test_maxpool.test_maxpool2d_creation()
    test_maxpool.test_maxpool2d_custom_stride()
    test_maxpool.test_maxpool2d_forward()
    test_maxpool.test_maxpool2d_odd_input()
    test_maxpool.test_maxpool2d_backward()
    test_maxpool.test_maxpool2d_gradient_values()

    test_avgpool = TestAvgPool2d()
    test_avgpool.test_avgpool2d_creation()
    test_avgpool.test_avgpool2d_custom_stride()
    test_avgpool.test_avgpool2d_forward()
    test_avgpool.test_avgpool2d_backward()
    test_avgpool.test_avgpool2d_gradient_values()

    test_flatten = TestFlatten()
    test_flatten.test_flatten_creation()
    test_flatten.test_flatten_forward()
    test_flatten.test_flatten_single_batch()
    test_flatten.test_flatten_backward()

    test_bn = TestBatchNorm2d()
    test_bn.test_batchnorm2d_creation()
    test_bn.test_batchnorm2d_forward()
    test_bn.test_batchnorm2d_eval_mode()
    test_bn.test_batchnorm2d_train_mode()
    test_bn.test_batchnorm2d_backward()

    test_drop = TestDropout2d()
    test_drop.test_dropout2d_creation()
    test_drop.test_dropout2d_train_mode()
    test_drop.test_dropout2d_eval_mode()
    test_drop.test_dropout2d_forward()

    test_int = TestCNNIntegration()
    test_int.test_conv_relu_pool()
    test_int.test_conv_batchnorm_relu()
    test_int.test_simple_cnn()
    test_int.test_end_to_end_backward()

    print("\n  all passed")