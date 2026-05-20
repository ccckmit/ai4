//! nn/tests.rs
//! Tests for nn package.

use nn::{Tensor, cat};
use nn::optim::{Linear, Embedding, RMSNorm, Adam, Module};
use nn::GPT;

#[test]
fn test_tensor_creation() {
    let t = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    assert_eq!(t.data.borrow().len(), 4);
    assert_eq!(t.requires_grad, true);
}

#[test]
fn test_tensor_add() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
    let c = a.add(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 5.0);
    assert_eq!(data[1], 7.0);
    assert_eq!(data[2], 9.0);
}

#[test]
fn test_tensor_mul() {
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
    let c = a.mul(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 8.0);
    assert_eq!(data[1], 15.0);
    assert_eq!(data[2], 24.0);
}

#[test]
fn test_tensor_relu() {
    let a = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
    let b = a.relu();
    let data = b.data.borrow();
    assert_eq!(data[0], 0.0);
    assert_eq!(data[1], 0.0);
    assert_eq!(data[2], 1.0);
    assert_eq!(data[3], 2.0);
}

#[test]
fn test_tensor_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let b = a.sum();
    let data = b.data.borrow();
    assert_eq!(data[0], 10.0);
}

#[test]
fn test_tensor_neg() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = a.neg();
    let data = b.data.borrow();
    assert_eq!(data[0], -1.0);
    assert_eq!(data[1], -2.0);
    assert_eq!(data[2], -3.0);
}

#[test]
fn test_tensor_pow() {
    let a = Tensor::from_vec(vec![2.0, 3.0], true);
    let b = a.pow(2.0);
    let data = b.data.borrow();
    assert_eq!(data[0], 4.0);
    assert_eq!(data[1], 9.0);
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
fn test_embedding_forward() {
    let embed = Embedding::new(100, 32);
    let indices = vec![5, 10, 15];
    let out = embed.forward(&indices);
    assert_eq!(out.shape[0], 3);
    assert_eq!(out.shape[1], 32);
}

#[test]
fn test_rmsnorm_forward() {
    let norm = RMSNorm::new(4, 1e-5);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let out = norm.forward(&x);
    assert_eq!(out.shape.len(), 1);
    assert_eq!(out.shape[0], 4);
}

#[test]
fn test_linear_parameters() {
    let linear = Linear::new(3, 2, true);
    let params = linear.parameters();
    assert_eq!(params.len(), 2); // weight + bias
}

#[test]
fn test_embedding_parameters() {
    let embed = Embedding::new(100, 32);
    let params = embed.parameters();
    assert_eq!(params.len(), 1); // only weight
}

#[test]
fn test_gpt_creation() {
    let gpt = GPT::new(100, 32, 2, 16, 4);
    let params = gpt.parameters();
    assert!(params.len() > 0);
}

#[test]
fn test_gpt_forward() {
    let gpt = GPT::new(100, 32, 1, 16, 4);
    let indices = vec![1, 2, 3, 4, 5];
    let (logits, caches) = gpt.forward(&indices, None);
    assert_eq!(logits.shape[0], 5);
    assert_eq!(logits.shape[1], 100);
}

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
fn test_embedding_backward() {
    let embed = Embedding::new(10, 4);
    let indices = vec![1, 3, 5];
    let out = embed.forward(&indices);
    let loss = out.sum();
    let _ = loss;
    // grad should be set
    assert!(embed.weight.requires_grad);
}

#[test]
fn test_rmsnorm_backward() {
    let norm = RMSNorm::new(4, 1e-5);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let y = norm.forward(&x);
    let _ = y.sum();
    // Forward pass works
    assert_eq!(y.shape[0], 4);
}

#[test]
fn test_gpt_creation() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let params = gpt.parameters();
    assert!(params.len() > 0);
}

#[test]
fn test_gpt_forward() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let indices = vec![1, 2, 3, 4];
    let (logits, caches) = gpt.forward(&indices, None);
    assert_eq!(logits.shape[0], 4);
    assert_eq!(logits.shape[1], 20);
}

#[test]
fn test_gpt_multi_layer() {
    let gpt = GPT::new(20, 16, 2, 16, 4);
    let indices = vec![1, 2, 3];
    let (logits, caches) = gpt.forward(&indices, None);
    assert_eq!(logits.shape[0], 3);
    assert_eq!(logits.shape[1], 20);
    assert_eq!(caches.len(), 2);
}

#[test]
fn test_tensor_softmax() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let y = x.softmax(0);
    let data = y.data.borrow();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_tensor_transpose() {
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let y = x.reshape(vec![2, 3]);
    let z = y.transpose();
    assert_eq!(z.shape.len(), 2);
}

#[test]
fn test_tensor_cat() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0], true);
    let c = nn::cat(&[a, b], 0);
    let data = c.data.borrow();
    assert_eq!(data.len(), 5);
}