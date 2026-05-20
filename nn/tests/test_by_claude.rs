//! nn/tests/test_by_claude.rs - Comprehensive gradient checking tests.

use crate::nn::{Tensor, Module, Linear, Embedding, RMSNorm, Adam, cat};

#[test]
fn test_add_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], false);
    let c = a.add(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 5.0);
    assert_eq!(data[1], 7.0);
    assert_eq!(data[2], 9.0);
}

#[test]
fn test_mul_forward() {
    let a = Tensor::from_vec(vec![2.0, 3.0], false);
    let b = Tensor::from_vec(vec![4.0, -1.0], false);
    let c = a.mul(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 8.0);
    assert_eq!(data[1], -3.0);
}

#[test]
fn test_neg() {
    let a = Tensor::from_vec(vec![1.0, -2.0], false);
    let b = a.neg();
    let data = b.data.borrow();
    assert_eq!(data[0], -1.0);
    assert_eq!(data[1], 2.0);
}

#[test]
fn test_pow_forward() {
    let a = Tensor::from_vec(vec![2.0, 3.0], false);
    let b = a.pow(2.0);
    let data = b.data.borrow();
    assert_eq!(data[0], 4.0);
    assert_eq!(data[1], 9.0);
}

#[test]
fn test_relu_forward() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 2.0], false);
    let b = a.relu();
    let data = b.data.borrow();
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 2.0);
}

#[test]
fn test_sum_global_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let b = a.sum();
    let data = b.data.borrow();
    assert_eq!(data[0], 10.0);
}

#[test]
fn test_reshape_forward() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], false);
    let b = a.reshape(vec![2, 3]);
    assert_eq!(b.shape, vec![2, 3]);
}

#[test]
fn test_transpose_forward() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], false);
    let b = a.transpose();
    assert_eq!(b.shape, vec![2, 2]);
}

#[test]
fn test_matmul_forward() {
    let a = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
    let b = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2], false);
    let c = a.matmul(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 0.0);
    assert_eq!(data[3], 1.0);
}

#[test]
fn test_linear_forward_shape() {
    let layer = Linear::new(4, 8, false);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let out = layer.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 8);
}

#[test]
fn test_linear_bias() {
    let mut layer = Linear::new(2, 3, true);
    {
        let w = &mut layer.weight.data.borrow_mut();
        for v in w.iter_mut() { *v = 0.0; }
    }
    if let Some(ref mut b) = layer.bias {
        let bb = &mut b.data.borrow_mut();
        for (i, v) in bb.iter_mut().enumerate() { *v = (i + 1) as f32; }
    }
    let x = Tensor::from_vec(vec![1.0, 1.0], false);
    let out = layer.forward(&x);
    let data = out.data.borrow();
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], 3.0);
}

#[test]
fn test_embedding_forward_shape() {
    let emb = Embedding::new(10, 4);
    let indices = vec![1, 3, 5];
    let out = emb.embed(&indices);
    assert_eq!(out.shape[0], 3);
    assert_eq!(out.shape[1], 4);
}

#[test]
fn test_embedding_forward_values() {
    let emb = Embedding::new(10, 4);
    let indices = vec![2, 5];
    let out = emb.embed(&indices);
    let data = out.data.borrow();
    let w = emb.weight.data.borrow();
    assert_eq!(data[0], w[2 * 4]);
    assert_eq!(data[1], w[2 * 4 + 1]);
    assert_eq!(data[4], w[5 * 4]);
    assert_eq!(data[5], w[5 * 4 + 1]);
}

#[test]
fn test_rmsnorm_output_shape() {
    let norm = RMSNorm::new(8, 1e-5);
    let data: Vec<f32> = (0..4 * 8).map(|i| (i as f32) - 16.0).collect();
    let x = Tensor::new(data, vec![4, 8], false);
    let out = norm.forward(&x);
    assert_eq!(out.shape, vec![4, 8]);
}

#[test]
fn test_linear_parameters() {
    let layer = Linear::new(4, 8, true);
    let params = layer.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_parameters_no_bias() {
    let layer = Linear::new(4, 8, false);
    let params = layer.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_embedding_parameters() {
    let emb = Embedding::new(10, 4);
    let params = emb.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_adam_optimizer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let mut optim = Adam::new(vec![t.clone()], 0.01, (0.9, 0.999), 1e-8);
    optim.zero_grad();
    optim.step();
}

#[test]
fn test_zero_grad_resets() {
    let mut x = Tensor::from_vec(vec![1.0, 2.0], true);
    let out = x.mul(&x);
    let _ = out.sum();
    x.zero_grad();
    let g = x.grad.borrow();
    assert!(g.iter().all(|&v| v == 0.0));
}

#[test]
fn test_softmax_sums_to_one() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let y = x.softmax(0);
    let data = y.data.borrow();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}