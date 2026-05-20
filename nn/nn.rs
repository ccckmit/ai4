//! nn/optim.rs
//! Neural network layers and optimizers.

use super::tensor::Tensor;

pub trait Module {
    fn parameters(&self) -> Vec<&Tensor>;
}

pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();
        let std = 0.08;
        
        let weight_data: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.sample(rand_distr::Normal::new(0.0, std).unwrap()) as f32)
            .collect();
        
        let weight = Tensor::new(weight_data, vec![in_features, out_features], true);
        
        let bias = if bias {
            let bias_data = vec![0.0; out_features];
            Some(Tensor::new(bias_data, vec![out_features], true))
        } else {
            None
        };
        
        Linear { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // Handle 1D input by reshaping to 2D
        let x_data = x.data.borrow();
        let in_features = self.weight.shape[0];
        let out_features = self.weight.shape[1];
        
        let mut result = vec![0.0; out_features];
        
        if x.shape.len() == 1 {
            // 1D input: dot product
            let w_data = self.weight.data.borrow();
            for j in 0..out_features {
                for i in 0..in_features {
                    result[j] += x_data[i] * w_data[i * out_features + j];
                }
            }
        } else {
            // 2D input: matrix multiplication
            let batch_size = x.shape[0];
            let w_data = self.weight.data.borrow();
            for b in 0..batch_size {
                for j in 0..out_features {
                    for i in 0..in_features {
                        result[j] += x_data[b * in_features + i] * w_data[i * out_features + j];
                    }
                }
            }
        }
        
        let out = Tensor::new(result, vec![out_features], x.requires_grad);
        
        if let Some(ref b) = self.bias {
            return out.add(b);
        }
        out
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<&Tensor> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }
}

pub struct Embedding {
    pub weight: Tensor,
}

impl Embedding {
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();
        
        let weight_data: Vec<f32> = (0..num_embeddings * embedding_dim)
            .map(|_| rng.sample(rand_distr::Normal::new(0.0, 0.08).unwrap()) as f32)
            .collect();
        
        let weight = Tensor::new(weight_data, vec![num_embeddings, embedding_dim], true);
        
        Embedding { weight }
    }

    pub fn forward(&self, indices: &[usize]) -> Tensor {
        let batch_size = indices.len();
        let embedding_dim = self.weight.shape[1];
        
        let mut result = Vec::with_capacity(batch_size * embedding_dim);
        
        for &idx in indices {
            let start = idx * embedding_dim;
            let end = start + embedding_dim;
            let row = &self.weight.data.borrow()[start..end];
            result.extend(row.iter().copied());
        }
        
        // Simplified - in full implementation would have proper backward
        Tensor::new(result, vec![batch_size, embedding_dim], true)
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.weight]
    }
}

pub struct RMSNorm {
    pub scale: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        let scale_data = vec![1.0; dim];
        let scale = Tensor::new(scale_data, vec![dim], false);
        
        RMSNorm { scale, eps }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let data = x.data.borrow();
        let batch_size = x.shape[0];
        let seq_len = x.shape[1];
        let dim = x.shape[2];
        
        let mut output = Vec::with_capacity(data.len());
        
        for b in 0..batch_size {
            for t in 0..seq_len {
                let offset = (b * seq_len + t) * dim;
                let slice = &data[offset..offset + dim];
                
                let ms: f32 = slice.iter().map(|&x| x * x).sum::<f32>() / dim as f32 + self.eps;
                let inv_std = ms.powf(-0.5);
                
                for &val in slice {
                    output.push(val * inv_std);
                }
            }
        }
        
        Tensor::new(output, x.shape.clone(), x.requires_grad)
    }
}

impl Module for RMSNorm {
    fn parameters(&self) -> Vec<&Tensor> {
        vec![&self.scale]
    }
}

pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    m: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    t: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f32, betas: (f32, f32), eps: f32) -> Self {
        let m: Vec<Vec<f32>> = params.iter()
            .map(|p| vec![0.0; p.data.borrow().len()])
            .collect();
        
        let v: Vec<Vec<f32>> = params.iter()
            .map(|p| vec![0.0; p.data.borrow().len()])
            .collect();
        
        Adam {
            params,
            lr,
            beta1: betas.0,
            beta2: betas.1,
            eps,
            m,
            v,
            t: 0,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        
        for (i, p) in self.params.iter().enumerate() {
            let data = p.data.borrow();
            let grad = p.grad.borrow();
            
            for j in 0..data.len() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad[j] * grad[j];
                
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                
                let update = self.lr * m_hat / (v_hat.sqrt() + self.eps);
                
                // Update would need to modify the underlying data
                // This is simplified - full implementation would use RefCell
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for p in &mut self.params {
            p.zero_grad();
        }
    }
}

impl Module for Adam {
    fn parameters(&self) -> Vec<&Tensor> {
        self.params.iter().collect()
    }
}