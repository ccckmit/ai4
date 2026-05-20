//! nn/tests/test_tensor.rs - Tensor autodiff tests.

use crate::nn::Tensor;

#[test]
fn test_tensor_creation() {
    let t = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
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
fn test_tensor_matmul() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
    let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2], true);
    let c = a.matmul(&b);
    let data = c.data.borrow();
    assert_eq!(data[0], 19.0);
    assert_eq!(data[1], 22.0);
    assert_eq!(data[2], 43.0);
    assert_eq!(data[3], 50.0);
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
fn test_tensor_softmax() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3], true);
    let y = x.softmax(1);
    let data = y.data.borrow();
    let sum: f32 = data.iter().sum();
    assert!((sum - 1.0).abs() < 0.01);
}

#[test]
fn test_tensor_transpose() {
    let x = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
    let y = x.transpose();
    assert_eq!(y.shape, vec![3, 2]);
}

#[test]
fn test_tensor_cat() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0], true);
    let c = crate::nn::cat(&[a, b], 0);
    let data = c.data.borrow();
    assert_eq!(data.len(), 5);
}

#[test]
fn test_tensor_neg() {
    let a = Tensor::from_vec(vec![1.0, -2.0, 3.0], true);
    let b = a.neg();
    let data = b.data.borrow();
    assert_eq!(data[0], -1.0);
    assert_eq!(data[1], 2.0);
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
fn test_tensor_sum() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], true);
    let b = a.sum();
    let data = b.data.borrow();
    assert_eq!(data[0], 10.0);
}

#[test]
fn test_tensor_reshape() {
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], true);
    let b = a.reshape(vec![2, 3]);
    assert_eq!(b.shape, vec![2, 3]);
}

#[test]
fn test_tensor_zeros() {
    let t = Tensor::zeros(&[3, 4]);
    assert_eq!(t.data.borrow().len(), 12);
    assert_eq!(t.shape, vec![3, 4]);
}

#[test]
fn test_tensor_ones() {
    let t = Tensor::ones(&[2, 3]);
    let data = t.data.borrow();
    assert!(data.iter().all(|&v| v == 1.0));
}