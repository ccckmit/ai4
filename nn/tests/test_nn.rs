//! nn/tests/test_nn.rs - Module, Linear, Embedding, RMSNorm, Adam tests.

use crate::nn::{Tensor, Module, Linear, Embedding, RMSNorm, Adam};

#[test]
fn test_module_parameters() {
    let linear = Linear::new(3, 4, true);
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_no_bias() {
    let linear = Linear::new(3, 4, false);
    let params = linear.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_linear_with_bias() {
    let linear = Linear::new(3, 4, true);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let out = linear.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 4);
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_forward() {
    let linear = Linear::new(3, 2, true);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], false);
    let out = linear.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 2);
}

#[test]
fn test_embedding() {
    let embed = Embedding::new(100, 32);
    let indices = vec![5, 10, 15];
    let out = embed.embed(&indices);
    assert_eq!(out.shape[0], 3);
    assert_eq!(out.shape[1], 32);
}

#[test]
fn test_embedding_parameters() {
    let embed = Embedding::new(100, 32);
    let params = embed.parameters();
    assert_eq!(params.len(), 1);
}

#[test]
fn test_embedding_backward() {
    let embed = Embedding::new(10, 4);
    let indices = vec![1, 3, 5];
    let out = embed.embed(&indices);
    let _loss = out.sum();
    assert!(embed.weight.requires_grad);
}

#[test]
fn test_rmsnorm() {
    let norm = RMSNorm::new(4, 1e-5);
    let data: Vec<f32> = (0..2 * 4).map(|i| i as f32 - 4.0).collect();
    let x = Tensor::new(data, vec![2, 4], true);
    let y = norm.forward(&x);
    assert_eq!(y.shape, vec![2, 4]);
}

#[test]
fn test_adam_optimizer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let mut optim = Adam::new(vec![t], 0.01, (0.9, 0.999), 1e-8);
    optim.zero_grad();
    optim.step();
}

#[test]
fn test_linear_parameters() {
    let linear = Linear::new(3, 2, true);
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_backward() {
    let linear = Linear::new(2, 1, true);
    let x = Tensor::from_vec(vec![1.0, 2.0], true);
    let y = linear.forward(&x);
    let _loss = y.sum();
    assert!(linear.weight.requires_grad);
}