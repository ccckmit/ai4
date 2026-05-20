//! nn/mnist/train.rs - MNIST training using simple MLP

use ai4::nn::{Tensor, Module, Linear};
use ai4::nn::datasets::{load_mnist, DataLoader};

struct MNISTNet {
    fc1: Linear,
    fc2: Linear,
}

impl MNISTNet {
    fn new() -> Self {
        MNISTNet {
            fc1: Linear::new(28 * 28, 128, true),
            fc2: Linear::new(128, 10, true),
        }
    }
}

impl Module for MNISTNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.reshape(vec![x.data.borrow().len() / (28 * 28), 28 * 28]);
        let x = x.relu();
        let x = self.fc1.forward(&x);
        let x = x.relu();
        self.fc2.forward(&x)
    }
}

fn train() {
    println!("Loading MNIST dataset...");
    
    let dataset = load_mnist("./data");
    println!("Dataset: {} samples", dataset.count());
    
    let mut loader = DataLoader::new(dataset, 64, true);
    
    let model = MNISTNet::new();
    
    let epochs = 1;
    for epoch in 0..epochs {
        let mut total = 0;
        let mut correct = 0;
        let mut batch_count = 0;
        
        while let Some((images, labels)) = loader.batch() {
            if images.is_empty() { break; }
            
            let batch_size = images.len();
            
            let mut x_data = Vec::new();
            for img in &images {
                x_data.extend(img.data.borrow().iter().cloned());
            }
            let x = Tensor::new(x_data, vec![batch_size, 1, 28, 28], true);
            
            let logits = model.forward(&x);
            let logits_data = logits.data.borrow();
            
            let batch_size = images.len();
            let num_classes = 10;
            let expected = batch_size * num_classes;
            
            if logits_data.len() != expected {
                print!("skip\r");
                continue;
            }
            
            let targets: Vec<usize> = labels.iter().map(|&l| l as usize % num_classes).collect();
            let _loss = logits.cross_entropy(&targets);
            
            for i in 0..batch_size {
                let start = i * num_classes;
                let end = start + num_classes;
                let slice = &logits_data[start..end];
                let mut max_idx = 0;
                let mut max_val = f32::NEG_INFINITY;
                for (j, &v) in slice.iter().enumerate() {
                    if v > max_val {
                        max_val = v;
                        max_idx = j;
                    }
                }
                if max_idx == targets[i] {
                    correct += 1;
                }
            }
            total += batch_size;
            batch_count += 1;
            
            print!("Epoch {} Batch {}\r", epoch + 1, batch_count);
        }
        
        let accuracy = if total > 0 { 100.0 * correct as f64 / total as f64 } else { 0.0 };
        println!("\nEpoch {} Accuracy: {:.2}%", epoch + 1, accuracy);
        
        loader = DataLoader::new(load_mnist("./data"), 64, true);
    }
    
    println!("Training complete!");
}

fn main() {
    train();
}