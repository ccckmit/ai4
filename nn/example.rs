//! nn/example.rs - Example of using neural network

use ai4::{Tensor, Linear, Embedding, Module};

fn main() {
    println!("=== Tensor Example ===");
    
    // Create tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let b = Tensor::from_vec(vec![4.0, 5.0, 6.0], true);
    
    // Operations
    let c = a.add(&b);
    println!("a + b = {:?}", c.data.borrow());
    
    let d = a.mul(&b);
    println!("a * b = {:?}", d.data.borrow());
    
    // ReLU
    let x = Tensor::from_vec(vec![-1.0, 0.0, 1.0, 2.0], true);
    let y = x.relu();
    println!("relu([-1, 0, 1, 2]) = {:?}", y.data.borrow());
    
    println!("\n=== Linear Layer Example ===");
    
    let linear = Linear::new(3, 2, true);
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], true);
    let out = linear.forward(&x);
    println!("Linear(3 -> 2): output shape = {:?}", out.shape);
    
    println!("\n=== Embedding Example ===");
    
    let embed = Embedding::new(100, 32);
    let indices = vec![5, 10, 15, 20];
    let out = embed.embed(&indices);
    println!("Embedding(100, 32): output shape = {:?}", out.shape);
}