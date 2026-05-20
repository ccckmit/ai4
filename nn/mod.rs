//! nn - DIY Neural Network framework with autograd.

pub mod tensor;
pub mod optim;
pub mod gpt;

pub use tensor::{Tensor, cat};
pub use optim::{Module, Linear, Embedding, RMSNorm, Adam};
pub use gpt::GPT;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tensor_add() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
        let c = a.add(&b);
        assert_eq!(c.data.borrow().len(), 3);
    }
    
    #[test]
    fn test_tensor_mul() {
        let a = Tensor::from_vec(vec![2.0, 3.0, 4.0], true);
        let b = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let c = a.mul(&b);
        assert_eq!(c.data.borrow().len(), 3);
    }
    
    #[test]
    fn test_relu() {
        let a = Tensor::from_vec(vec![-1.0, 2.0, -3.0, 4.0], true);
        let b = a.relu();
        let data = b.data.borrow();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 2.0);
    }
    
    #[test]
    fn test_linear() {
        let linear = Linear::new(3, 2, true);
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let out = linear.forward(&x);
        assert_eq!(out.shape, vec![2]);
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
    fn test_softmax() {
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
        let y = x.softmax(0);
        let data = y.data.borrow();
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_gpt_creation() {
        let gpt = GPT::new(10, 8, 1, 8, 2);
        let params = gpt.parameters();
        assert!(params.len() > 0);
    }
}