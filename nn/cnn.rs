use super::tensor::{Tensor, SimpleRng};

pub trait Module {
    fn forward(&self, x: &Tensor) -> Tensor;
    fn parameters(&self) -> Vec<Tensor> { vec![] }
}

fn im2col(
    data: &[f64],
    n: usize, c: usize, h: usize, w: usize,
    kh: usize, kw: usize,
    stride: usize, pad: usize,
) -> Vec<f64> {
    let out_h = (h + 2 * pad).saturating_sub(kh) / stride + 1;
    let out_w = (w + 2 * pad).saturating_sub(kw) / stride + 1;
    let h2 = h + 2 * pad;
    let w2 = w + 2 * pad;

    let mut padded = vec![0.0; n * c * h2 * w2];
    for b in 0..n {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    padded[b * c * h2 * w2 + ch * h2 * w2 + (y + pad) * w2 + (x + pad)] =
                        data[b * c * h * w + ch * h * w + y * w + x];
                }
            }
        }
    }

    let c_kh_kw = c * kh * kw;
    let n_oh_ow = n * out_h * out_w;
    let mut col = vec![0.0; c_kh_kw * n_oh_ow];

    for ch in 0..c {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = ch * kh * kw + ky * kw + kx;
                for b in 0..n {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let cow = b * out_h * out_w + oh * out_w + ow;
                            let y = ky + oh * stride;
                            let x = kx + ow * stride;
                            let pad_idx = b * c * h2 * w2 + ch * h2 * w2 + y * w2 + x;
                            col[row * n_oh_ow + cow] = padded[pad_idx];
                        }
                    }
                }
            }
        }
    }
    col
}

fn col2im(
    col: &[f64],
    n: usize, c: usize, h: usize, w: usize,
    kh: usize, kw: usize,
    stride: usize, pad: usize,
) -> Vec<f64> {
    let out_h = (h + 2 * pad).saturating_sub(kh) / stride + 1;
    let out_w = (w + 2 * pad).saturating_sub(kw) / stride + 1;
    let h2 = h + 2 * pad;
    let w2 = w + 2 * pad;

    let mut padded = vec![0.0; n * c * h2 * w2];
    let n_oh_ow = n * out_h * out_w;

    for ch in 0..c {
        for ky in 0..kh {
            for kx in 0..kw {
                let row = ch * kh * kw + ky * kw + kx;
                for b in 0..n {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let cow = b * out_h * out_w + oh * out_w + ow;
                            let y = ky + oh * stride;
                            let x = kx + ow * stride;
                            let pad_idx = b * c * h2 * w2 + ch * h2 * w2 + y * w2 + x;
                            padded[pad_idx] += col[row * n_oh_ow + cow];
                        }
                    }
                }
            }
        }
    }

    let mut out = vec![0.0; n * c * h * w];
    for b in 0..n {
        for ch in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out[b * c * h * w + ch * h * w + y * w + x] =
                        padded[b * c * h2 * w2 + ch * h2 * w2 + (y + pad) * w2 + (x + pad)];
                }
            }
        }
    }
    out
}

pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

impl Conv2d {
    pub fn new(in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, use_bias: bool) -> Self {
        let mut rng = SimpleRng::new(12345);
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f64).sqrt();
        let w_size = out_channels * in_channels * kernel_size * kernel_size;
        let mut w = vec![0.0; w_size];
        for i in 0..w_size {
            w[i] = (rng.next_f64() * 2.0 - 1.0) * scale;
        }
        let weight = Tensor::new(w, vec![out_channels, in_channels, kernel_size, kernel_size], true);

        let bias = if use_bias {
            Some(Tensor::new(vec![0.0; out_channels], vec![out_channels], true))
        } else {
            None
        };

        Self { in_channels, out_channels, kernel_size, stride, padding, weight, bias }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];

        let out_h = (h + 2 * self.padding).saturating_sub(self.kernel_size) / self.stride + 1;
        let out_w = (w + 2 * self.padding).saturating_sub(self.kernel_size) / self.stride + 1;
        let oh_ow = out_h * out_w;
        let n_out_h_out_w = n * oh_ow;
        
        let k_per_oc = self.in_channels * self.kernel_size * self.kernel_size;
        let x_data = x.data().clone();
        let x_col = im2col(&x_data, n, c, h, w, self.kernel_size, self.kernel_size, self.stride, self.padding);
        let w_row = self.weight.data().clone();
        
        let mut out_data = vec![0.0; n * self.out_channels * oh_ow];

        for oc in 0..self.out_channels {
            let oc_offset = oc * k_per_oc;
            for col_row in 0..k_per_oc {
                let w_val = w_row[oc_offset + col_row];
                let col_row_offset = col_row * n_out_h_out_w;
                for b in 0..n {
                    let b_col_offset = b * oh_ow;
                    let b_out_offset = (b * self.out_channels + oc) * oh_ow;
                    for spatial in 0..oh_ow {
                        out_data[b_out_offset + spatial] += w_val * x_col[col_row_offset + b_col_offset + spatial];
                    }
                }
            }
            if let Some(ref bias) = self.bias {
                let b_val = bias.data()[oc];
                for b in 0..n {
                    let b_out_offset = (b * self.out_channels + oc) * oh_ow;
                    for spatial in 0..oh_ow {
                        out_data[b_out_offset + spatial] += b_val;
                    }
                }
            }
        }

        let rg = x.requires_grad() || self.weight.requires_grad();
        let out = Tensor::new(out_data, vec![n, self.out_channels, out_h, out_w], rg);

        if rg {
            out.push_prev(x.clone());
            out.push_prev(self.weight.clone());
            if let Some(ref bias) = self.bias {
                out.push_prev(bias.clone());
            }

            let x_c = x.clone();
            let w_c = self.weight.clone();
            let b_c = self.bias.clone();
            let out_c = out.clone();
            let oc_size = self.out_channels;
            let ks = self.kernel_size;
            let pad = self.padding;
            let stride = self.stride;
            
            out.set_backward_fn(Box::new(move || {
                let dout = out_c.grad().clone();

                if x_c.requires_grad() {
                    let mut dcol = vec![0.0; k_per_oc * n_out_h_out_w];
                    let w_data = w_c.data().clone();
                    for oc in 0..oc_size {
                        let oc_offset = oc * k_per_oc;
                        for col_row in 0..k_per_oc {
                            let w_val = w_data[oc_offset + col_row];
                            let col_row_offset = col_row * n_out_h_out_w;
                            for b in 0..n {
                                let b_col_offset = b * oh_ow;
                                let b_out_offset = (b * oc_size + oc) * oh_ow;
                                for spatial in 0..oh_ow {
                                    dcol[col_row_offset + b_col_offset + spatial] += w_val * dout[b_out_offset + spatial];
                                }
                            }
                        }
                    }
                    let dx_data = col2im(&dcol, n, c, h, w, ks, ks, stride, pad);
                    let mut x_mut = x_c.grad_mut();
                    for i in 0..x_mut.len() {
                        x_mut[i] += dx_data[i];
                    }
                }

                if w_c.requires_grad() {
                    let mut w_mut = w_c.grad_mut();
                    for oc in 0..oc_size {
                        let oc_offset = oc * k_per_oc;
                        for col_row in 0..k_per_oc {
                            let mut sum = 0.0;
                            let col_row_offset = col_row * n_out_h_out_w;
                            for b in 0..n {
                                let b_col_offset = b * oh_ow;
                                let b_out_offset = (b * oc_size + oc) * oh_ow;
                                for spatial in 0..oh_ow {
                                    sum += dout[b_out_offset + spatial] * x_col[col_row_offset + b_col_offset + spatial];
                                }
                            }
                            w_mut[oc_offset + col_row] += sum;
                        }
                    }
                }

                if let Some(ref bias) = b_c {
                    if bias.requires_grad() {
                        let mut b_mut = bias.grad_mut();
                        for oc in 0..oc_size {
                            let mut sum = 0.0;
                            for b in 0..n {
                                let b_out_offset = (b * oc_size + oc) * oh_ow;
                                for spatial in 0..oh_ow {
                                    sum += dout[b_out_offset + spatial];
                                }
                            }
                            b_mut[oc] += sum;
                        }
                    }
                }
            }));
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut v = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            v.push(b.clone());
        }
        v
    }
}

pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        Self { 
            kernel_size, 
            stride: stride.unwrap_or(kernel_size) 
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        
        let out_h = (h.saturating_sub(self.kernel_size)) / self.stride + 1;
        let out_w = (w.saturating_sub(self.kernel_size)) / self.stride + 1;
        let mut out_data = vec![0.0; n * c * out_h * out_w];
        let x_data = x.data().clone();
        
        for b in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut max_val = std::f64::NEG_INFINITY;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let y = oh * self.stride + kh;
                                let x_idx = ow * self.stride + kw;
                                if y < h && x_idx < w {
                                    let val = x_data[b * c * h * w + ch * h * w + y * w + x_idx];
                                    if val > max_val {
                                        max_val = val;
                                    }
                                }
                            }
                        }
                        out_data[b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] = max_val;
                    }
                }
            }
        }
        
        let rg = x.requires_grad();
        let out = Tensor::new(out_data, vec![n, c, out_h, out_w], rg);
        
        if rg {
            out.push_prev(x.clone());
            let out_c = out.clone();
            let x_c = x.clone();
            let ks = self.kernel_size;
            let stride = self.stride;
            
            out.set_backward_fn(Box::new(move || {
                let dout = out_c.grad().clone();
                let mut x_mut = x_c.grad_mut();
                
                for b in 0..n {
                    for ch in 0..c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let mut max_val = std::f64::NEG_INFINITY;
                                let mut max_h = 0;
                                let mut max_w = 0;
                                for kh in 0..ks {
                                    for kw in 0..ks {
                                        let y = oh * stride + kh;
                                        let x_idx = ow * stride + kw;
                                        if y < h && x_idx < w {
                                            let val = x_data[b * c * h * w + ch * h * w + y * w + x_idx];
                                            if val > max_val {
                                                max_val = val;
                                                max_h = y;
                                                max_w = x_idx;
                                            }
                                        }
                                    }
                                }
                                let d = dout[b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow];
                                x_mut[b * c * h * w + ch * h * w + max_h * w + max_w] += d;
                            }
                        }
                    }
                }
            }));
        }
        out
    }
}

pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        Self { 
            kernel_size, 
            stride: stride.unwrap_or(kernel_size) 
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        
        let out_h = (h.saturating_sub(self.kernel_size)) / self.stride + 1;
        let out_w = (w.saturating_sub(self.kernel_size)) / self.stride + 1;
        let mut out_data = vec![0.0; n * c * out_h * out_w];
        let k_area = (self.kernel_size * self.kernel_size) as f64;
        let x_data = x.data().clone();
        
        for b in 0..n {
            for ch in 0..c {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let mut sum = 0.0;
                        for kh in 0..self.kernel_size {
                            for kw in 0..self.kernel_size {
                                let y = oh * self.stride + kh;
                                let x_idx = ow * self.stride + kw;
                                if y < h && x_idx < w {
                                    sum += x_data[b * c * h * w + ch * h * w + y * w + x_idx];
                                }
                            }
                        }
                        out_data[b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] = sum / k_area;
                    }
                }
            }
        }
        
        let rg = x.requires_grad();
        let out = Tensor::new(out_data, vec![n, c, out_h, out_w], rg);
        
        if rg {
            out.push_prev(x.clone());
            let out_c = out.clone();
            let x_c = x.clone();
            let ks = self.kernel_size;
            let stride = self.stride;
            
            out.set_backward_fn(Box::new(move || {
                let dout = out_c.grad().clone();
                let mut x_mut = x_c.grad_mut();
                
                for b in 0..n {
                    for ch in 0..c {
                        for oh in 0..out_h {
                            for ow in 0..out_w {
                                let g = dout[b * c * out_h * out_w + ch * out_h * out_w + oh * out_w + ow] / k_area;
                                for kh in 0..ks {
                                    for kw in 0..ks {
                                        let y = oh * stride + kh;
                                        let x_idx = ow * stride + kw;
                                        if y < h && x_idx < w {
                                            x_mut[b * c * h * w + ch * h * w + y * w + x_idx] += g;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }));
        }
        out
    }
}

pub struct Flatten {}

impl Flatten {
    pub fn new() -> Self { Self {} }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let batch_size = shape[0];
        let flat_dim: usize = shape[1..].iter().product();
        x.reshape(vec![batch_size, flat_dim])
    }
}

pub struct BatchNorm2d {
    pub num_channels: usize,
    pub eps: f64,
    pub momentum: f64,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: std::cell::RefCell<Vec<f64>>,
    pub running_var: std::cell::RefCell<Vec<f64>>,
    pub training: bool,
}

impl BatchNorm2d {
    pub fn new(num_channels: usize, eps: f64, momentum: f64) -> Self {
        Self {
            num_channels,
            eps,
            momentum,
            weight: Tensor::new(vec![1.0; num_channels], vec![num_channels], true),
            bias: Tensor::new(vec![0.0; num_channels], vec![num_channels], true),
            running_mean: std::cell::RefCell::new(vec![0.0; num_channels]),
            running_var: std::cell::RefCell::new(vec![1.0; num_channels]),
            training: true,
        }
    }

    pub fn eval(&mut self) { self.training = false; }
    pub fn train(&mut self) { self.training = true; }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let shape = x.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let mut out_data = vec![0.0; n * c * h * w];
        let x_data = x.data().clone();
        
        let w_data = self.weight.data().clone();
        let b_data = self.bias.data().clone();
        
        let mut running_mean = self.running_mean.borrow_mut();
        let mut running_var = self.running_var.borrow_mut();
        let eps = self.eps;
        
        let mut batch_mean = vec![0.0; c];
        let mut batch_var = vec![0.0; c];
        let is_training = self.training;

        if is_training {
            let numel_per_chan = (n * h * w) as f64;
            for ch in 0..c {
                let mut mean = 0.0;
                for b in 0..n {
                    for y in 0..h {
                        for x_idx in 0..w {
                            mean += x_data[b * c * h * w + ch * h * w + y * w + x_idx];
                        }
                    }
                }
                mean /= numel_per_chan;
                batch_mean[ch] = mean;
                running_mean[ch] = (1.0 - self.momentum) * running_mean[ch] + self.momentum * mean;
                
                let mut var = 0.0;
                for b in 0..n {
                    for y in 0..h {
                        for x_idx in 0..w {
                            let diff = x_data[b * c * h * w + ch * h * w + y * w + x_idx] - mean;
                            var += diff * diff;
                        }
                    }
                }
                var /= numel_per_chan;
                batch_var[ch] = var;
                running_var[ch] = (1.0 - self.momentum) * running_var[ch] + self.momentum * var;
                
                let inv_std = 1.0 / (var + eps).sqrt();
                for b in 0..n {
                    for y in 0..h {
                        for x_idx in 0..w {
                            let idx = b * c * h * w + ch * h * w + y * w + x_idx;
                            out_data[idx] = w_data[ch] * (x_data[idx] - mean) * inv_std + b_data[ch];
                        }
                    }
                }
            }
        } else {
            for ch in 0..c {
                let mean = running_mean[ch];
                let var = running_var[ch];
                let inv_std = 1.0 / (var + eps).sqrt();
                for b in 0..n {
                    for y in 0..h {
                        for x_idx in 0..w {
                            let idx = b * c * h * w + ch * h * w + y * w + x_idx;
                            out_data[idx] = w_data[ch] * (x_data[idx] - mean) * inv_std + b_data[ch];
                        }
                    }
                }
            }
        }

        let rg = x.requires_grad() || self.weight.requires_grad() || self.bias.requires_grad();
        let out = Tensor::new(out_data, shape.clone(), rg);
        
        if rg && is_training {
            out.push_prev(x.clone());
            out.push_prev(self.weight.clone());
            out.push_prev(self.bias.clone());
            let out_c = out.clone();
            let x_c = x.clone();
            let w_c = self.weight.clone();
            let b_c = self.bias.clone();
            
            out.set_backward_fn(Box::new(move || {
                let dout = out_c.grad().clone();
                let mut x_mut = x_c.grad_mut();
                let mut w_mut = w_c.grad_mut();
                let mut b_mut = b_c.grad_mut();
                
                for ch in 0..c {
                    let mut grad_gamma = 0.0;
                    let mut grad_beta = 0.0;
                    let iv = 1.0 / (batch_var[ch] + eps).sqrt();
                    for b in 0..n {
                        for y in 0..h {
                            for x_idx in 0..w {
                                let idx = b * c * h * w + ch * h * w + y * w + x_idx;
                                let d = dout[idx];
                                grad_gamma += d * (x_data[idx] - batch_mean[ch]) * iv;
                                grad_beta += d;
                            }
                        }
                    }
                    w_mut[ch] += grad_gamma;
                    b_mut[ch] += grad_beta;
                    
                    let w_gamma = w_data[ch];
                    // Simplified backward
                    for b in 0..n {
                        for y in 0..h {
                            for x_idx in 0..w {
                                let idx = b * c * h * w + ch * h * w + y * w + x_idx;
                                x_mut[idx] += dout[idx] * w_gamma * iv;
                            }
                        }
                    }
                }
            }));
        }
        out
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

pub struct Dropout2d {
    pub p: f64,
    pub training: bool,
}

impl Dropout2d {
    pub fn new(p: f64) -> Self {
        Self { p, training: true }
    }
    pub fn eval(&mut self) { self.training = false; }
    pub fn train(&mut self) { self.training = true; }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return x.clone();
        }
        let shape = x.shape();
        let len = x.data().len();
        let mut rng = SimpleRng::new(12345);
        let mut mask = vec![0.0; len];
        let mut out_data = vec![0.0; len];
        let scale = 1.0 / (1.0 - self.p);
        let in_data = x.data().clone();
        for i in 0..len {
            if rng.next_f64() >= self.p {
                mask[i] = scale;
                out_data[i] = in_data[i] * scale;
            }
        }
        let rg = x.requires_grad();
        let out = Tensor::new(out_data, shape, rg);
        if rg {
            out.push_prev(x.clone());
            let out_c = out.clone();
            let x_c = x.clone();
            out.set_backward_fn(Box::new(move || {
                let dout = out_c.grad().clone();
                let mut x_mut = x_c.grad_mut();
                for i in 0..len {
                    x_mut[i] += dout[i] * mask[i];
                }
            }));
        }
        out
    }
}
