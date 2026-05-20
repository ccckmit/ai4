//! nn/datasets.rs
//! MNIST dataset loader via Python/torch subprocess.
//! Calls `python -c "..."` to get raw data, parses into Rust Tensor.

use std::process::Command;
use crate::nn::Tensor;

pub struct DataLoader {
    dataset: Dataset,
    batch_size: usize,
    shuffle: bool,
    indices: Vec<usize>,
    pos: usize,
}

pub struct Dataset {
    pub images: Vec<Tensor>,
    pub labels: Vec<usize>,
    pub count: usize,
}

pub fn load_mnist(root: &str) -> Dataset {
    let script = format!(
        r#"
import sys
import pickle
try:
    from torchvision import datasets
    from torchvision.transforms import Compose, ToTensor, Normalize
    transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
    train_ds = datasets.MNIST('{root}', train=True, download=True, transform=transform)
    test_ds  = datasets.MNIST('{root}', train=False, download=True, transform=transform)
    data = []
    for img, label in train_ds:
        arr = img.squeeze().numpy()
        data.append((arr.tobytes(), arr.shape, label))
    for img, label in test_ds:
        arr = img.squeeze().numpy()
        data.append((arr.tobytes(), arr.shape, label + 10))
    sys.stdout.write(str(len(data)))
except Exception as e:
    sys.stderr.write(str(e))
    sys.exit(1)
"#
    );

    let output = Command::new("python")
        .arg("-c")
        .arg(&script)
        .output()
        .expect("Failed to execute python for MNIST download");

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        eprintln!("MNIST download failed: {}", stderr);
    }

    let count = String::from_utf8_lossy(&output.stdout).trim().parse::<usize>().unwrap_or(0);

    let script2 = format!(
        r#"
import sys
import pickle
import numpy as np
from torchvision import datasets
from torchvision.transforms import Compose, ToTensor, Normalize
transform = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])
train_ds = datasets.MNIST('{root}', train=True, download=True, transform=transform)
test_ds  = datasets.MNIST('{root}', train=False, download=True, transform=transform)

out = []
for img, label in train_ds:
    arr = img.squeeze().numpy()
    out.append(arr.tobytes().hex() + ',' + str(label) + ',' + str(arr.shape[0]) + ',' + str(arr.shape[1]))
for img, label in test_ds:
    arr = img.squeeze().numpy()
    out.append(arr.tobytes().hex() + ',' + str(label + 10) + ',' + str(arr.shape[0]) + ',' + str(arr.shape[1]))
sys.stdout.write(';'.join(out))
"#
    );

    let output2 = Command::new("python")
        .arg("-c")
        .arg(&script2)
        .output()
        .expect("Failed to execute python for MNIST data");

    let raw = String::from_utf8_lossy(&output2.stdout);
    let mut images = Vec::new();
    let mut labels = Vec::new();

    for entry in raw.split(';') {
        let parts: Vec<&str> = entry.split(',').collect();
        if parts.len() < 4 {
            continue;
        }
        let hex_data = parts[0];
        let label = parts[1].parse::<usize>().unwrap_or(0);
        let h = parts[2].parse::<usize>().unwrap_or(28);
        let w = parts[3].parse::<usize>().unwrap_or(28);

        if let Ok(bytes) = hex::decode(hex_data) {
            let data: Vec<f32> = bytes
                .chunks(4)
                .map(|chunk| {
                    let mut f = [0u8; 4];
                    f.copy_from_slice(chunk);
                    f32::from_le_bytes(f)
                })
                .collect();

            let shape = vec![1, h, w];
            images.push(Tensor::new(data, shape, false));
            labels.push(label);
        }
    }

    let count = images.len();
    Dataset { images, labels, count }
}

impl DataLoader {
    pub fn new(dataset: Dataset, batch_size: usize, shuffle: bool) -> Self {
        let indices: Vec<usize> = (0..dataset.count).collect();
        DataLoader {
            dataset,
            batch_size,
            shuffle,
            indices,
            pos: 0,
        }
    }

    pub fn batch(&mut self) -> Option<(Vec<Tensor>, Vec<usize>)> {
        if self.pos >= self.dataset.count {
            return None;
        }

        if self.pos == 0 && self.shuffle {
            use rand::Rng;
            let mut rng = rand::thread_rng();
            for i in (1..self.indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                self.indices.swap(i, j);
            }
        }

        let start = self.pos;
        let end = (self.pos + self.batch_size).min(self.dataset.count);
        self.pos = end;

        let batch_images: Vec<Tensor> = self.indices[start..end]
            .iter()
            .map(|&i| self.dataset.images[i].clone())
            .collect();
        let batch_labels: Vec<usize> = self.indices[start..end]
            .iter()
            .map(|&i| self.dataset.labels[i])
            .collect();

        Some((batch_images, batch_labels))
    }

    pub fn len(&self) -> usize {
        (self.dataset.count + self.batch_size - 1) / self.batch_size
    }
}

impl Dataset {
    pub fn count(&self) -> usize {
        self.count
    }
}

pub struct Compose {
    transforms: Vec<Box<dyn Transform>>,
}

pub trait Transform: Send + Sync {
    fn transform(&self, img: &[f32], h: usize, w: usize) -> Vec<f32>;
}

pub struct Grayscale;
pub struct Normalize {
    mean: f32,
    std: f32,
}
pub struct RandomRotation {
    _degrees: i32,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Compose { transforms }
    }
}

impl Transform for Normalize {
    fn transform(&self, img: &[f32], _h: usize, _w: usize) -> Vec<f32> {
        img.iter().map(|x| (x - self.mean) / self.std).collect()
    }
}

impl Transform for Grayscale {
    fn transform(&self, img: &[f32], _h: usize, _w: usize) -> Vec<f32> {
        img.to_vec()
    }
}

impl Transform for RandomRotation {
    fn transform(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        let _size = h * w;
        img.to_vec()
    }
}

impl Compose {
    pub fn apply(&self, img: &[f32], h: usize, w: usize) -> Vec<f32> {
        let mut out = img.to_vec();
        for t in &self.transforms {
            out = t.transform(&out, h, w);
        }
        out
    }
}