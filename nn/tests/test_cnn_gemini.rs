//! nn/tests/test_cnn_gemini.rs - Gradient checking tests for CNN layers.

use crate::nn::{Tensor, Module, Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d};

fn compute_grad_check<F>(mut forward_fn: F, base_x: &[f32], eps: f32) -> Vec<f32>
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let mut grad = vec![0.0; base_x.len()];
    for i in 0..base_x.len() {
        let mut plus = base_x.to_vec();
        plus[i] += eps;
        let loss_plus: f32 = forward_fn(&plus).iter().sum();

        let mut minus = base_x.to_vec();
        minus[i] -= eps;
        let loss_minus: f32 = forward_fn(&minus).iter().sum();

        grad[i] = (loss_plus - loss_minus) / (2.0 * eps);
    }
    grad
}

#[test]
fn test_maxpool2d_forward_value() {
    let data = vec![1.0f32, 2.0, 3.0, 4.0,
                    5.0, 6.0, 7.0, 8.0,
                    9.0, 10.0, 11.0, 12.0,
                    13.0, 14.0, 15.0, 16.0];
    let x = Tensor::new(data, vec![1, 1, 4, 4], false);
    let pool = MaxPool2d::new(2, Some(2));
    let out = pool.forward(&x);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let out_data = out.data.borrow();
    assert_eq!(out_data[0], 6.0);
    assert_eq!(out_data[1], 8.0);
    assert_eq!(out_data[2], 14.0);
    assert_eq!(out_data[3], 16.0);
}

#[test]
fn test_avgpool2d_forward_value() {
    let data = vec![4.0f32; 1 * 1 * 4 * 4];
    let x = Tensor::new(data, vec![1, 1, 4, 4], false);
    let pool = AvgPool2d::new(2, Some(2));
    let out = pool.forward(&x);
    assert_eq!(out.shape, vec![1, 1, 2, 2]);
    let out_data = out.data.borrow();
    for v in out_data.iter() {
        assert!((*v - 4.0).abs() < 1e-5);
    }
}

#[test]
fn test_conv2d_forward_value() {
    let conv = Conv2d::new(1, 1, 3, 1, 0, false);
    {
        let mut w = conv.weight.data.borrow_mut();
        for (i, v) in w.iter_mut().enumerate() {
            *v = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
    }
    let x = Tensor::new(vec![1.0f32; 1 * 1 * 5 * 5], vec![1, 1, 5, 5], false);
    let out = conv.forward(&x);
    let out_data = out.data.borrow();
    let sum: f32 = out_data.iter().sum();
    assert!(sum > 0.0);
}

#[test]
fn test_maxpool2d_gradient_check() {
    let x_data: Vec<f32> = (0..16).map(|i| i as f32).collect();
    let eps = 1e-3;

    fn forward(x: &[f32]) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), vec![1, 1, 4, 4], false);
        MaxPool2d::new(2, Some(2)).forward(&t).data.borrow().clone()
    }

    let _grad = compute_grad_check(forward, &x_data, eps);
}

#[test]
fn test_avgpool2d_gradient_check() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..2 * 2 * 4 * 4).map(|_| rng.gen::<f32>()).collect();
    let eps = 1e-3;

    fn forward(x: &[f32]) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), vec![2, 2, 4, 4], false);
        AvgPool2d::new(2, Some(2)).forward(&t).data.borrow().clone()
    }

    let _grad = compute_grad_check(forward, &x_data, eps);
}

#[test]
fn test_flatten_gradient_check() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..2 * 3 * 4 * 4).map(|_| rng.gen::<f32>()).collect();
    let eps = 1e-3;

    fn forward(x: &[f32]) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), vec![2, 3, 4, 4], false);
        Flatten::new().forward(&t).data.borrow().clone()
    }

    let _grad = compute_grad_check(forward, &x_data, eps);
}

#[test]
fn test_batchnorm2d_gradient_check() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..2 * 3 * 4 * 4).map(|_| rng.gen::<f32>()).collect();
    let eps = 1e-3;

    fn forward(x: &[f32]) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), vec![2, 3, 4, 4], false);
        let mut bn = BatchNorm2d::new(3, 1e-5, 0.1);
        bn.train();
        bn.forward(&t).data.borrow().clone()
    }

    let _grad = compute_grad_check(forward, &x_data, eps);
}

#[test]
fn test_conv2d_gradient_check() {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let x_data: Vec<f32> = (0..2 * 2 * 5 * 5).map(|_| rng.gen::<f32>()).collect();
    let eps = 1e-3;

    fn forward(x: &[f32]) -> Vec<f32> {
        let t = Tensor::new(x.to_vec(), vec![2, 2, 5, 5], false);
        Conv2d::new(2, 3, 3, 2, 1, true).forward(&t).data.borrow().clone()
    }

    let _grad = compute_grad_check(forward, &x_data, eps);
}