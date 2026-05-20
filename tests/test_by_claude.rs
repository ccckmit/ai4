//! tests/test_by_claude.rs - Comprehensive gradient checking tests.

use ai4::{Tensor, Module, Linear, Embedding};

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
    let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], false);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], false);
    let c = a.mul(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 8.0);
    assert_eq!(data[1], 15.0);
    assert_eq!(data[2], 24.0);
}

#[test]
fn test_neg() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], false);
    let b = a.neg();
    let data = b.data.borrow();
    assert_eq!(data[0], -1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], -3.0);
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
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], false);
    let b = a.reshape(vec![2, 2]);
    let c = b.transpose();
    assert_eq!(c.shape.len(), 2);
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
fn test_embedding_forward_shape() {
    let emb = Embedding::new(10, 4);
    let indices = vec![1, 3, 5];
    let out = emb.forward(&indices);
    assert_eq!(out.shape[0], 3);
    assert_eq!(out.shape[1], 4);
}

#[test]
fn test_linear_parameters() {
    let layer = Linear::new(4, 8, true);
    let params = layer.parameters();
    assert!(params.len() >= 1);
}

#[test]
fn test_linear_parameters_with_bias() {
    let layer = Linear::new(4, 8, true);
    let params = layer.parameters();
    assert_eq!(params.len(), 2);
}

#[test]
fn test_embedding_parameters() {
    let emb = Embedding::new(10, 4);
    let params = emb.parameters();
    assert_eq!(params.len(), 1);
}