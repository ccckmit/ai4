//! nn/tests/test_cnn.rs - CNN layer tests.

use crate::nn::{Tensor, Module, Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d};

#[test]
fn test_conv2d_creation() {
    let conv = Conv2d::new(3, 16, 3, 1, 0, true);
    assert_eq!(conv.in_channels, 3);
    assert_eq!(conv.out_channels, 16);
    assert_eq!(conv.kernel_size, 3);
}

#[test]
fn test_conv2d_forward() {
    let conv = Conv2d::new(3, 8, 3, 1, 0, true);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = conv.forward(&x);
    let expected_h = (10 - 3) / 1 + 1;
    let expected_w = (10 - 3) / 1 + 1;
    assert_eq!(out.shape, vec![2, 8, expected_h, expected_w]);
}

#[test]
fn test_conv2d_with_padding() {
    let conv = Conv2d::new(3, 8, 3, 1, 1, true);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = conv.forward(&x);
    assert_eq!(out.shape, vec![2, 8, 10, 10]);
}

#[test]
fn test_conv2d_with_stride() {
    let conv = Conv2d::new(3, 8, 3, 2, 0, true);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = conv.forward(&x);
    let expected_h = (10 - 3) / 2 + 1;
    let expected_w = (10 - 3) / 2 + 1;
    assert_eq!(out.shape, vec![2, 8, expected_h, expected_w]);
}

#[test]
fn test_conv2d_no_bias() {
    let conv = Conv2d::new(3, 8, 3, 1, 0, false);
    assert!(conv.bias.is_none());
}

#[test]
fn test_conv2d_single_channel() {
    let conv = Conv2d::new(1, 1, 3, 1, 0, false);
    let data = vec![0.0f32; 1 * 1 * 10 * 10];
    let x = Tensor::new(data, vec![1, 1, 10, 10], false);
    let out = conv.forward(&x);
    assert_eq!(out.shape, vec![1, 1, 8, 8]);
}

#[test]
fn test_maxpool2d_creation() {
    let pool = MaxPool2d::new(2, None);
    assert_eq!(pool.kernel_size, 2);
    assert_eq!(pool.stride, 2);
}

#[test]
fn test_maxpool2d_custom_stride() {
    let pool = MaxPool2d::new(2, Some(1));
    assert_eq!(pool.stride, 1);
}

#[test]
fn test_maxpool2d_forward() {
    let pool = MaxPool2d::new(2, None);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = pool.forward(&x);
    assert_eq!(out.shape, vec![2, 3, 5, 5]);
}

#[test]
fn test_maxpool2d_odd_input() {
    let pool = MaxPool2d::new(3, Some(2));
    let data = vec![0.0f32; 1 * 1 * 7 * 7];
    let x = Tensor::new(data, vec![1, 1, 7, 7], false);
    let out = pool.forward(&x);
    let expected = (7 - 3) / 2 + 1;
    assert_eq!(out.shape, vec![1, 1, expected, expected]);
}

#[test]
fn test_avgpool2d_creation() {
    let pool = AvgPool2d::new(2, None);
    assert_eq!(pool.kernel_size, 2);
    assert_eq!(pool.stride, 2);
}

#[test]
fn test_avgpool2d_forward() {
    let pool = AvgPool2d::new(2, None);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = pool.forward(&x);
    assert_eq!(out.shape, vec![2, 3, 5, 5]);
}

#[test]
fn test_flatten_creation() {
    let flat = Flatten::new();
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);
    let out = flat.forward(&x);
    assert_eq!(out.shape, vec![2, 300]);
}

#[test]
fn test_flatten_single_batch() {
    let flat = Flatten::new();
    let data = vec![0.0f32; 1 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![1, 3, 10, 10], false);
    let out = flat.forward(&x);
    assert_eq!(out.shape, vec![1, 300]);
}

#[test]
fn test_batchnorm2d_creation() {
    let bn = BatchNorm2d::new(16, 1e-5, 0.1);
    assert_eq!(bn.num_channels, 16);
    assert!(bn.training);
}

#[test]
fn test_batchnorm2d_forward() {
    let bn = BatchNorm2d::new(8, 1e-5, 0.1);
    let data = vec![0.0f32; 4 * 8 * 10 * 10];
    let x = Tensor::new(data, vec![4, 8, 10, 10], false);
    let out = bn.forward(&x);
    assert_eq!(out.shape, vec![4, 8, 10, 10]);
}

#[test]
fn test_batchnorm2d_eval_mode() {
    let mut bn = BatchNorm2d::new(8, 1e-5, 0.1);
    bn.eval();
    assert!(!bn.training);
}

#[test]
fn test_batchnorm2d_train_mode() {
    let mut bn = BatchNorm2d::new(8, 1e-5, 0.1);
    bn.eval();
    bn.train();
    assert!(bn.training);
}

#[test]
fn test_dropout2d_creation() {
    let drop = Dropout2d::new(0.5);
    assert!((drop.p - 0.5).abs() < 1e-6);
    assert!(drop.training);
}

#[test]
fn test_dropout2d_eval_mode() {
    let mut drop = Dropout2d::new(0.5);
    drop.eval();
    assert!(!drop.training);
}

#[test]
fn test_conv_relu_pool() {
    let conv = Conv2d::new(3, 8, 3, 1, 1, true);
    let pool = MaxPool2d::new(2, None);
    let data = vec![0.0f32; 2 * 3 * 10 * 10];
    let x = Tensor::new(data, vec![2, 3, 10, 10], false);

    let out = conv.forward(&x);
    let out = out.relu();
    let out = pool.forward(&out);
    assert_eq!(out.shape, vec![2, 8, 5, 5]);
}

#[test]
fn test_simple_cnn() {
    let conv1 = Conv2d::new(1, 8, 3, 1, 1, true);
    let pool1 = MaxPool2d::new(2, None);
    let conv2 = Conv2d::new(8, 16, 3, 1, 1, true);
    let pool2 = MaxPool2d::new(2, None);
    let flat = Flatten::new();

    let data = vec![0.0f32; 4 * 1 * 28 * 28];
    let x = Tensor::new(data, vec![4, 1, 28, 28], false);

    let out = conv1.forward(&x);
    let out = out.relu();
    let out = pool1.forward(&out);
    assert_eq!(out.shape, vec![4, 8, 14, 14]);

    let out = conv2.forward(&out);
    let out = out.relu();
    let out = pool2.forward(&out);
    assert_eq!(out.shape, vec![4, 16, 7, 7]);

    let out = flat.forward(&out);
    assert_eq!(out.shape, vec![4, 784]);
}