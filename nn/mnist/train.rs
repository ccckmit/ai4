//! nn/mnist/train.rs - MNIST training using CNN

use ai4::nn::{Tensor, Module, Conv2d, MaxPool2d, Flatten, Linear, Adam};
use std::fs::File;
use std::io::{Write, Read};

pub struct Dataset {
    pub images: Vec<f64>,
    pub labels: Vec<usize>,
}

impl Dataset {
    pub fn count(&self) -> usize {
        self.labels.len()
    }
}

pub fn load_mnist(dir: &str) -> Dataset {
    let mut img_file = File::open(format!("{}/train-images-idx3-ubyte", dir))
        .or_else(|_| File::open(format!("{}/train-images.idx3-ubyte", dir)))
        .expect("MNIST images not found");
    let mut lbl_file = File::open(format!("{}/train-labels-idx1-ubyte", dir))
        .or_else(|_| File::open(format!("{}/train-labels.idx1-ubyte", dir)))
        .expect("MNIST labels not found");

    let mut buf4 = [0u8; 4];
    img_file.read_exact(&mut buf4).unwrap();
    img_file.read_exact(&mut buf4).unwrap();
    let num_images = u32::from_be_bytes(buf4) as usize;
    img_file.read_exact(&mut buf4).unwrap();
    img_file.read_exact(&mut buf4).unwrap();

    let mut img_data = vec![0u8; num_images * 28 * 28];
    img_file.read_exact(&mut img_data).unwrap();

    lbl_file.read_exact(&mut buf4).unwrap();
    lbl_file.read_exact(&mut buf4).unwrap();
    let num_labels = u32::from_be_bytes(buf4) as usize;

    let mut lbl_data = vec![0u8; num_labels];
    lbl_file.read_exact(&mut lbl_data).unwrap();

    Dataset {
        images: img_data.iter().map(|&x| x as f64 / 255.0).collect(),
        labels: lbl_data.iter().map(|&x| x as usize).collect(),
    }
}

pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    idx: usize,
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, _shuffle: bool) -> Self {
        Self { dataset, batch_size, idx: 0 }
    }
    pub fn batch(&mut self) -> Option<(Vec<Vec<f64>>, Vec<usize>)> {
        if self.idx >= self.dataset.count() {
            return None;
        }
        let end = (self.idx + self.batch_size).min(self.dataset.count());
        let mut imgs = Vec::new();
        let mut lbls = Vec::new();
        for i in self.idx..end {
            let start = i * 28 * 28;
            imgs.push(self.dataset.images[start..start + 28 * 28].to_vec());
            lbls.push(self.dataset.labels[i]);
        }
        self.idx = end;
        Some((imgs, lbls))
    }
}

struct MNISTNet {
    conv1: Conv2d,
    conv2: Conv2d,
    pool1: MaxPool2d,
    pool2: MaxPool2d,
    flatten: Flatten,
    fc1: Linear,
    fc2: Linear,
}

impl MNISTNet {
    fn new() -> Self {
        use ai4::nn::tensor::SimpleRng;
        let mut rng = SimpleRng::new(12345);
        MNISTNet {
            conv1: Conv2d::new(1, 32, 3, 1, 0, true),
            conv2: Conv2d::new(32, 64, 3, 1, 0, true),
            pool1: MaxPool2d::new(2, None),
            pool2: MaxPool2d::new(2, None),
            flatten: Flatten::new(),
            fc1: Linear::new(64 * 5 * 5, 128, true, &mut rng),
            fc2: Linear::new(128, 10, true, &mut rng),
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.conv1.parameters();
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

impl Module for MNISTNet {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x = x.relu();
        let x = self.conv1.forward(&x);
        let x = self.pool1.forward(&x);
        let x = x.relu();
        let x = self.conv2.forward(&x);
        let x = self.pool2.forward(&x);
        let x = self.flatten.forward(&x);
        let x = x.relu();
        let x = self.fc1.forward(&x);
        self.fc2.forward(&x)
    }
}

fn save_model(model: &MNISTNet, path: &str) {
    if let Ok(mut file) = File::create(path) {
        write!(file, "{{").unwrap();
        let params = model.parameters();
        for (i, p) in params.iter().enumerate() {
            if i > 0 { write!(file, ",").unwrap(); }
            write!(file, "\"param_{}\": [", i).unwrap();
            let data = p.data();
            for (j, &val) in data.iter().enumerate() {
                if j > 0 { write!(file, ",").unwrap(); }
                write!(file, "{}", val).unwrap();
            }
            write!(file, "]").unwrap();
        }
        write!(file, "}}").unwrap();
        println!("Model saved to {}", path);
    }
}

fn train() {
    println!("Loading MNIST dataset...");
    
    // We only take a subset for quick testing if preferred, but here we just
    // load all and do a standard run.
    let dataset = load_mnist("./data");
    println!("Dataset: {} samples", dataset.count());
    let subset_size = 1000.min(dataset.count());
    let mut loader = DataLoader::new(dataset, 64, true);
    
    let model = MNISTNet::new();
    let mut optimizer = Adam::new(model.parameters(), 0.001);
    
    let epochs = 3;
    for epoch in 0..epochs {
        let mut total = 0;
        let mut correct = 0;
        let mut batch_count = 0;
        
        while let Some((images, labels)) = loader.batch() {
            if images.is_empty() { break; }
            if total >= subset_size { break; } // stop at 1000 images per epoch like TS
            
            let batch_size = images.len();
            
            let mut x_data = Vec::new();
            for img in &images {
                x_data.extend(img.iter().cloned());
            }
            let x = Tensor::new(x_data, vec![batch_size, 1, 28, 28], true);
            
            optimizer.zero_grad();
            let logits = model.forward(&x);
            let logits_data = logits.data();
            
            let num_classes = 10;
            let expected = batch_size * num_classes;
            if logits_data.len() != expected {
                continue;
            }
            
            let targets: Vec<usize> = labels.iter().map(|&l| l as usize % num_classes).collect();
            let loss = logits.cross_entropy(&targets);
            loss.backward();
            optimizer.step();
            
            let loss_val = loss.data()[0];
            
            for i in 0..batch_size {
                let start = i * num_classes;
                let end = start + num_classes;
                let slice = &logits_data[start..end];
                let mut max_idx = 0;
                let mut max_val = f64::NEG_INFINITY;
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
            
            println!("Epoch {} Batch {} Loss: {:.4}", epoch + 1, batch_count - 1, loss_val);
        }
        
        let accuracy = if total > 0 { 100.0 * correct as f64 / total as f64 } else { 0.0 };
        println!("Epoch {} Accuracy: {:.2}%", epoch + 1, accuracy);
        
        loader = DataLoader::new(load_mnist("./data"), 64, true);
    }
    
    let _ = std::fs::create_dir_all("nn/mnist");
    save_model(&model, "nn/mnist/model.json");
    println!("Training complete!");
}

fn main() {
    train();
}