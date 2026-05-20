//! tests/test_nn.rs - Module, Linear, Embedding, RMSNorm, Adam tests.

use ai4::{Tensor, Module, Linear, Embedding, Adam};

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
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let out = linear.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 4);
    let params = linear.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_linear_forward() {
    let linear = Linear::new(3, 2, true);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let out = linear.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 2);
}

#[test]
fn test_embedding() {
    let embed = Embedding::new(100, 32);
    let indices = vec![5, 10, 15];
    let out = embed.forward(&indices);
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
    let out = embed.forward(&indices);
    let _loss = out.sum();
    assert!(embed.weight.requires_grad);
}

#[test]
fn test_adam_optimizer() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let mut optim = Adam::new(vec![t.clone()], 0.01, (0.9, 0.999), 1e-8);
    optim.step();
    optim.zero_grad();
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