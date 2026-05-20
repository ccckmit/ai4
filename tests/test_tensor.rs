//! tests/test_tensor.rs - Tensor autodiff tests.

use ai4::Tensor;

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
    let c = ai4::cat(&[a, b], 0);
    let data = c.data.borrow();
    assert_eq!(data.len(), 5);
}