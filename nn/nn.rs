//! nn/optim.rs
//! Neural network layers and optimizers.

use crate::tensor::Tensor; // 假設上一份檔案名為 tensor.rs

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    
    // 改為回傳 Tensor (也就是 Rc Handle)，這樣不會有生命週期 (Lifetime) 問題
    fn parameters(&self) -> Vec<Tensor> {
        vec![]
    }
}

/* =========================================
   Linear Layer (全連接層)
   ========================================= */
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> Self {
        let std = (2.0 / in_features as f32).sqrt(); // 建議使用 Kaiming 初始化
        
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::from_entropy();
        
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
}

impl Module for Linear {
    fn forward(&self, x: &Tensor) -> Tensor {
        // 使用 tensor.rs 中支援反向傳播的 matmul
        let mut out = x.matmul(&self.weight);
        
        if let Some(ref b) = self.bias {
            // 因為 tensor.rs 原本的 add 不支援 Broadcasting (廣播機制)，
            // 這裡我們手動註冊一個 Bias Add 計算圖節點來正確支援 2D [Batch, Out] + 1D [Out]
            out = broadcast_add_bias(&out, b);
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}

// 輔助函數：將 1D bias 加到 2D 矩陣上並註冊反向傳播
fn broadcast_add_bias(x: &Tensor, bias: &Tensor) -> Tensor {
    let x_b = x.inner.borrow();
    let bias_b = bias.inner.borrow();
    
    let batch_size = x_b.shape[0];
    let out_features = x_b.shape[1];
    
    let mut data = vec![0.0; x_b.data.len()];
    for i in 0..batch_size {
        for j in 0..out_features {
            data[i * out_features + j] = x_b.data[i * out_features + j] + bias_b.data[j];
        }
    }
    
    let requires_grad = x_b.requires_grad || bias_b.requires_grad;
    let out = Tensor::new(data, x_b.shape.clone(), requires_grad);
    out.inner.borrow_mut()._prev = vec![x.clone(), bias.clone()];
    
    if requires_grad {
        let x_c = x.clone();
        let b_c = bias.clone();
        let out_c = out.clone();
        
        out.inner.borrow_mut()._backward = Some(Box::new(move || {
            let out_g = out_c.inner.borrow().grad.clone();
            let mut x_g = x_c.inner.borrow_mut();
            let mut b_g = b_c.inner.borrow_mut();
            
            for i in 0..batch_size {
                for j in 0..out_features {
                    let grad = out_g[i * out_features + j];
                    if x_g.requires_grad { x_g.grad[i * out_features + j] += grad; }
                    // Bias 梯度是整個 Batch 加總
                    if b_g.requires_grad { b_g.grad[j] += grad; } 
                }
            }
        }));
    }
    out
}

/* =========================================
   Embedding Layer (嵌入層)
   ========================================= */
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
        
        Embedding {
            weight: Tensor::new(weight_data, vec![num_embeddings, embedding_dim], true),
        }
    }
}

impl Module for Embedding {
    fn forward(&self, x: &Tensor) -> Tensor {
        let indices: Vec<usize> = x.inner.borrow().data.iter().map(|&v| v as usize).collect();
        let w_b = self.weight.inner.borrow();
        let emb_dim = w_b.shape[1];
        let batch_size = indices.len();
        
        let mut data = Vec::with_capacity(batch_size * emb_dim);
        for &idx in &indices {
            let start = idx * emb_dim;
            data.extend_from_slice(&w_b.data[start..start + emb_dim]);
        }
        
        // 正確註冊計算圖與反向傳播 (Scatter Add)
        let out = Tensor::new(data, vec![batch_size, emb_dim], w_b.requires_grad);
        out.inner.borrow_mut()._prev = vec![self.weight.clone()];
        
        if w_b.requires_grad {
            let w_c = self.weight.clone();
            let out_c = out.clone();
            
            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad.clone();
                let mut w_g = w_c.inner.borrow_mut();
                
                for (i, &idx) in indices.iter().enumerate() {
                    let out_start = i * emb_dim;
                    let w_start = idx * emb_dim;
                    for j in 0..emb_dim {
                        w_g.grad[w_start + j] += out_g[out_start + j];
                    }
                }
            }));
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/* =========================================
   RMSNorm Layer
   ========================================= */
pub struct RMSNorm {
    pub scale: Tensor,
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(dim: usize, eps: f32) -> Self {
        RMSNorm {
            scale: Tensor::new(vec![1.0; dim], vec![dim], true), // scale 必須可訓練
            eps,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, x: &Tensor) -> Tensor {
        let x_b = x.inner.borrow();
        let scale_b = self.scale.inner.borrow();
        
        let dim = *x_b.shape.last().unwrap();
        let rows = x_b.data.len() / dim;
        
        let mut data = vec![0.0; x_b.data.len()];
        
        for i in 0..rows {
            let offset = i * dim;
            let slice = &x_b.data[offset..offset + dim];
            
            let ms: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + self.eps;
            let inv_std = ms.powf(-0.5);
            
            for j in 0..dim {
                // 正確做法：正規化後必須乘上 scale 參數
                data[offset + j] = slice[j] * inv_std * scale_b.data[j];
            }
        }
        
        let requires_grad = x_b.requires_grad || scale_b.requires_grad;
        let out = Tensor::new(data, x_b.shape.clone(), requires_grad);
        out.inner.borrow_mut()._prev = vec![x.clone(), self.scale.clone()];
        
        if requires_grad {
            let x_c = x.clone();
            let scale_c = self.scale.clone();
            let out_c = out.clone();
            let eps = self.eps;
            
            out.inner.borrow_mut()._backward = Some(Box::new(move || {
                let out_g = out_c.inner.borrow().grad.clone();
                let x_d = x_c.inner.borrow().data.clone();
                let s_d = scale_c.inner.borrow().data.clone();
                
                let mut x_g = x_c.inner.borrow_mut();
                let mut s_g = scale_c.inner.borrow_mut();
                
                for i in 0..rows {
                    let offset = i * dim;
                    let slice = &x_d[offset..offset + dim];
                    
                    let ms: f32 = slice.iter().map(|&v| v * v).sum::<f32>() / dim as f32 + eps;
                    let inv_std = ms.powf(-0.5);
                    let inv_std_3 = ms.powf(-1.5);
                    
                    let mut sum_dx_dot_x = 0.0;
                    for j in 0..dim {
                        let dy = out_g[offset + j];
                        sum_dx_dot_x += dy * s_d[j] * x_d[offset + j];
                        
                        if s_g.requires_grad {
                            s_g.grad[j] += dy * x_d[offset + j] * inv_std;
                        }
                    }
                    
                    if x_g.requires_grad {
                        for j in 0..dim {
                            let dy = out_g[offset + j];
                            let dx = (dy * s_d[j] * inv_std) - (x_d[offset + j] * sum_dx_dot_x * inv_std_3 / dim as f32);
                            x_g.grad[offset + j] += dx;
                        }
                    }
                }
            }));
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.scale.clone()]
    }
}

/* =========================================
   Adam Optimizer (不再實作 Module Trait)
   ========================================= */
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
        let m: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.inner.borrow().data.len()]).collect();
        let v: Vec<Vec<f32>> = params.iter().map(|p| vec![0.0; p.inner.borrow().data.len()]).collect();
        
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
            // 使用 borrow_mut 取出可變參考，真正更新資料！
            let mut inner = p.inner.borrow_mut();
            
            for j in 0..inner.data.len() {
                let grad = inner.grad[j];
                
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * grad;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * grad * grad;
                
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                
                // 真正在這裡把權重數據減去更新值！
                inner.data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}