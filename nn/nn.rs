// ============================================================
//  nn.rs  –  Layers and optimiser
// ============================================================
use super::tensor::{SimpleRng, Tensor};

// ── Linear ───────────────────────────────────────────────────
pub struct Linear {
    pub weight: Tensor, // [out, in]
    pub bias: Option<Tensor>,
}

impl Linear {
    pub fn new(in_f: usize, out_f: usize, use_bias: bool, rng: &mut SimpleRng) -> Self {
        let std = 0.08f64;
        let w: Vec<f64> = (0..out_f * in_f).map(|_| rng.randn() * std).collect();
        let weight = Tensor::from_slice(&w, &[out_f, in_f], true);
        let bias = if use_bias {
            Some(Tensor::new(vec![0.0; out_f], vec![out_f], true))
        } else {
            None
        };
        Self { weight, bias }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let nd = self.weight.shape().len();
        let wt = self.weight.transpose(nd - 2, nd - 1);
        let out = x.matmul(&wt);
        match &self.bias {
            Some(b) => out.add(b),
            None => out,
        }
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut v = vec![self.weight.clone()];
        if let Some(b) = &self.bias {
            v.push(b.clone());
        }
        v
    }
}

// ── Embedding ────────────────────────────────────────────────
pub struct Embedding {
    pub weight: Tensor, // [num_emb, dim]
}

impl Embedding {
    pub fn new(num_emb: usize, dim: usize, rng: &mut SimpleRng) -> Self {
        let std = 0.08f64;
        let w: Vec<f64> = (0..num_emb * dim).map(|_| rng.randn() * std).collect();
        Self {
            weight: Tensor::from_slice(&w, &[num_emb, dim], true),
        }
    }

    /// indices: Tensor [B, T] (integer values stored as f64)
    pub fn forward(&self, indices: &Tensor) -> Tensor {
        let idx_shape = indices.shape();
        let (batch, seq) = (idx_shape[0], idx_shape[1]);
        let ws = self.weight.shape();
        let (num_emb, dim) = (ws[0], ws[1]);
        let idx_data = indices.data().clone();
        let w_data = self.weight.data().clone();

        let mut out_data = vec![0.0f64; batch * seq * dim];
        for b in 0..batch {
            for t in 0..seq {
                let idx = (idx_data[b * seq + t].round() as usize).min(num_emb - 1);
                for j in 0..dim {
                    out_data[(b * seq + t) * dim + j] = w_data[idx * dim + j];
                }
            }
        }

        let out = Tensor::new(out_data, vec![batch, seq, dim], self.weight.requires_grad());
        out.push_prev(self.weight.clone());

        if self.weight.requires_grad() {
            let out_c = out.clone();
            let weight2 = self.weight.clone();
            out.set_backward_fn(Box::new(move || {
                let og = out_c.grad().clone();
                let mut wg = weight2.grad_mut();
                for b in 0..batch {
                    for t in 0..seq {
                        let idx = (idx_data[b * seq + t].round() as usize).min(num_emb - 1);
                        for j in 0..dim {
                            wg[idx * dim + j] += og[(b * seq + t) * dim + j];
                        }
                    }
                }
            }));
        }
        out
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

// ── RMSNorm ───────────────────────────────────────────────────
pub struct RMSNorm {
    pub scale: Tensor,
}

impl RMSNorm {
    pub fn new(dim: usize) -> Self {
        Self {
            scale: Tensor::new(vec![1.0; dim], vec![dim], false),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rms_norm().mul(&self.scale)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        if self.scale.requires_grad() {
            vec![self.scale.clone()]
        } else {
            vec![]
        }
    }
}

// ── Adam optimiser ────────────────────────────────────────────
pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    m: Vec<Vec<f64>>,
    v: Vec<Vec<f64>>,
    pub t: usize,
}

impl Adam {
    pub fn new(params: Vec<Tensor>, lr: f64) -> Self {
        let m: Vec<Vec<f64>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        let v: Vec<Vec<f64>> = params.iter().map(|p| vec![0.0; p.numel()]).collect();
        Self {
            params,
            lr,
            beta1: 0.85,
            beta2: 0.99,
            eps: 1e-8,
            m,
            v,
            t: 0,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        for (i, p) in self.params.iter().enumerate() {
            let grad = p.grad().clone();
            let mut data = p.data_mut();
            for j in 0..data.len() {
                let g = grad[j];
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g * g;
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    let diff = pred.sub(target);
    let sq = diff.mul(&diff);
    sq.mean_all()
}
