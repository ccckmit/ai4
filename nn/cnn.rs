//! nn/cnn.rs
//! Convolutional Neural Network layers.
//! Conv2d: 2D convolution layer
//! MaxPool2d: 2D max pooling
//! AvgPool2d: 2D average pooling
//! Flatten: flatten tensor for dense layers
//! BatchNorm2d: 2D Spatial Batch Normalization
//! Dropout2d: 2D Channel-wise Dropout

use crate::nn::Module;
use crate::nn::Tensor;
use rand::Rng;

pub struct Conv2d {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
    pub weight: Tensor,
    pub bias: Option<Tensor>,
}

pub struct MaxPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

pub struct AvgPool2d {
    pub kernel_size: usize,
    pub stride: usize,
}

pub struct Flatten;

pub struct BatchNorm2d {
    pub num_channels: usize,
    pub eps: f32,
    pub momentum: f32,
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Vec<f32>,
    pub running_var: Vec<f32>,
    pub training: bool,
}

pub struct Dropout2d {
    pub p: f32,
    pub training: bool,
}

fn get_im2col_indices(
    x_shape: &[usize],
    field_height: usize,
    field_width: usize,
    padding: usize,
    stride: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    let n = x_shape[0];
    let c = x_shape[1];
    let h = x_shape[2];
    let w = x_shape[3];

    let out_h = (h + 2 * padding - field_height) / stride + 1;
    let out_w = (w + 2 * padding - field_width) / stride + 1;

    let mut i0 = Vec::with_capacity(field_height * field_width * c);
    for _ in 0..c {
        for i in 0..field_height {
            for j in 0..field_width {
                i0.push(i * field_width + j);
            }
        }
    }

    let mut i1 = Vec::with_capacity(out_h * out_w);
    for i in 0..out_h {
        for j in 0..out_w {
            i1.push(i * stride);
        }
    }

    let mut j0 = Vec::with_capacity(field_height * field_width * c);
    for _ in 0..c {
        for i in 0..field_height {
            for j in 0..field_width {
                j0.push(j);
            }
        }
    }

    let mut j1 = Vec::with_capacity(out_h * out_w);
    for i in 0..out_h {
        for j in 0..out_w {
            j1.push(j * stride);
        }
    }

    let mut k = Vec::with_capacity(field_height * field_width * c);
    for c_i in 0..c {
        for _ in 0..field_height * field_width {
            k.push(c_i);
        }
    }

    (k, i1, j1)
}

fn im2col_indices(x: &[f32], x_shape: &[usize], kernel_size: usize, padding: usize, stride: usize) -> Vec<f32> {
    let n = x_shape[0];
    let c = x_shape[1];
    let h = x_shape[2];
    let w = x_shape[3];

    let padded_h = h + 2 * padding;
    let padded_w = w + 2 * padding;
    let mut x_padded = vec![0.0; n * c * padded_h * padded_w];

    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let dst_idx = ni * c * padded_h * padded_w + ci * padded_h * padded_w + (hi + padding) * padded_w + (wi + padding);
                    let src_idx = ni * c * h * w + ci * h * w + hi * w + wi;
                    x_padded[dst_idx] = x[src_idx];
                }
            }
        }
    }

    let (k, i1, j1) = get_im2col_indices(x_shape, kernel_size, kernel_size, padding, stride);
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;
    let patch_size = kernel_size * kernel_size * c;
    let num_patches = out_h * out_w * n;

    let mut cols = vec![0.0; patch_size * num_patches];

    for ni in 0..n {
        for patch_idx in 0..out_h * out_w {
            let i_offset = i1[patch_idx];
            let j_offset = j1[patch_idx];

            for ki in 0..patch_size {
                let c_idx = ki / (kernel_size * kernel_size);
                let h_idx = (ki % (kernel_size * kernel_size)) / kernel_size;
                let w_idx = ki % kernel_size;

                let h_pos = i_offset + h_idx;
                let w_pos = j_offset + w_idx;
                let src = ni * c * padded_h * padded_w + c_idx * padded_h * padded_w + h_pos * padded_w + w_pos;
                let dst = patch_idx * patch_size + ni * patch_size + ki;
                if src < x_padded.len() && dst < cols.len() {
                    cols[dst] = x_padded[src];
                }
            }
        }
    }

    cols
}

fn im2col_for_pool(x: &[f32], x_shape: &[usize], kernel_size: usize, stride: usize) -> Vec<f32> {
    let n = x_shape[0];
    let c = x_shape[1];
    let h = x_shape[2];
    let w = x_shape[3];

    let out_h = (h - kernel_size) / stride + 1;
    let out_w = (w - kernel_size) / stride + 1;
    let patch_size = kernel_size * kernel_size;
    let num_patches = n * c * out_h * out_w;

    let mut cols = vec![0.0; patch_size * num_patches];

    for ni in 0..n {
        for ci in 0..c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let patch_base = ((ni * c + ci) * out_h + oy) * out_w + ox;
                    for ky in 0..kernel_size {
                        for kx in 0..kernel_size {
                            let sy = oy * stride + ky;
                            let sx = ox * stride + kx;
                            let src = ni * c * h * w + ci * h * w + sy * w + sx;
                            let dst = patch_base * patch_size + ky * kernel_size + kx;
                            if src < x.len() {
                                cols[dst] = x[src];
                            }
                        }
                    }
                }
            }
        }
    }

    cols
}

fn col2im_indices(cols: &[f32], x_shape: &[usize], kernel_size: usize, padding: usize, stride: usize) -> Vec<f32> {
    let n = x_shape[0];
    let c = x_shape[1];
    let h = x_shape[2];
    let w = x_shape[3];

    let padded_h = h + 2 * padding;
    let padded_w = w + 2 * padding;

    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;
    let patch_size = kernel_size * kernel_size * c;

    let mut x_padded = vec![0.0; n * c * padded_h * padded_w];

    for ni in 0..n {
        for patch_idx in 0..out_h * out_w {
            let i_offset = patch_idx / out_w * stride;
            let j_offset = patch_idx % out_w * stride;

            for ki in 0..patch_size {
                let c_idx = ki / (kernel_size * kernel_size);
                let h_idx = (ki % (kernel_size * kernel_size)) / kernel_size;
                let w_idx = ki % kernel_size;

                let h_pos = i_offset + h_idx;
                let w_pos = j_offset + w_idx;

                let src = patch_idx * patch_size + ni * patch_size + ki;
                let dst = ni * c * padded_h * padded_w + c_idx * padded_h * padded_w + h_pos * padded_w + w_pos;

                if src < cols.len() && dst < x_padded.len() {
                    x_padded[dst] += cols[src];
                }
            }
        }
    }

    if padding == 0 {
        return x_padded;
    }

    let mut x_out = vec![0.0; n * c * h * w];
    for ni in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ni * c * padded_h * padded_w + ci * padded_h * padded_w + (hi + padding) * padded_w + (wi + padding);
                    let dst = ni * c * h * w + ci * h * w + hi * w + wi;
                    x_out[dst] = x_padded[src];
                }
            }
        }
    }

    x_out
}

impl Conv2d {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> Self {
        let scale = (2.0 / (in_channels * kernel_size * kernel_size) as f32).sqrt();
        let size = out_channels * in_channels * kernel_size * kernel_size;
        let data: Vec<f32> = (0..size)
            .map(|_| {
                let mut rng = rand::thread_rng();
                let r: f32 = rng.gen();
                (r - 0.5) * 2.0 * scale
            })
            .collect();

        let weight = Tensor::new(data, vec![out_channels, in_channels, kernel_size, kernel_size], true);

        let bias_tensor = if bias {
            Some(Tensor::zeros(&[out_channels]))
        } else {
            None
        };

        Conv2d {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            weight,
            bias: bias_tensor,
        }
    }
}

impl Module for Conv2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];

        let out_h = (h + 2 * self.padding - self.kernel_size) / self.stride + 1;
        let out_w = (w + 2 * self.padding - self.kernel_size) / self.stride + 1;

        let x_col = im2col_indices(&x.data.borrow(), &[n, c, h, w], self.kernel_size, self.padding, self.stride);

        let w_row_size = self.out_channels;
        let k_size = self.in_channels * self.kernel_size * self.kernel_size;
        let num_patches = out_h * out_w * n;

        let mut out_col = vec![0.0; w_row_size * num_patches];

        for out_c in 0..self.out_channels {
            for patch in 0..num_patches {
                let mut sum = 0.0f32;
                for k in 0..k_size {
                    let w_idx = out_c * k_size + k;
                    let x_idx = k * num_patches + patch;
                    if w_idx < self.weight.data.borrow().len() && x_idx < x_col.len() {
                        sum += self.weight.data.borrow()[w_idx] * x_col[x_idx];
                    }
                }
                out_col[out_c * num_patches + patch] = sum;
            }
        }

        if let Some(ref bias) = self.bias {
            for out_c in 0..self.out_channels {
                for patch in 0..num_patches {
                    let idx = out_c * num_patches + patch;
                    out_col[idx] += bias.data.borrow()[out_c];
                }
            }
        }

        let mut out_data = vec![0.0; n * self.out_channels * out_h * out_w];

        for ni in 0..n {
            for oc in 0..self.out_channels {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let patch_idx = ni * (self.out_channels * out_h * out_w) + oc * (out_h * out_w) + oy * out_w + ox;
                        let src = oc * (n * out_h * out_w) + ni * (out_h * out_w) + oy * out_w + ox;
                        if src < out_col.len() {
                            out_data[patch_idx] = out_col[src];
                        }
                    }
                }
            }
        }

        let shape = vec![n, self.out_channels, out_h, out_w];
        Tensor::new(out_data, shape, x.requires_grad)
    }
}

impl MaxPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        MaxPool2d {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
        }
    }
}

impl Module for MaxPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];

        let out_h = (h - self.kernel_size) / self.stride + 1;
        let out_w = (w - self.kernel_size) / self.stride + 1;

        let mut x_reshaped = vec![0.0; n * c * h * w];
        for i in 0..x_reshaped.len() {
            if i < x.data.borrow().len() {
                x_reshaped[i] = x.data.borrow()[i];
            }
        }

        let x_col = im2col_for_pool(&x_reshaped, &[n * c, 1, h, w], self.kernel_size, self.stride);
        let patch_size = self.kernel_size * self.kernel_size;
        let num_patches = n * c * out_h * out_w;

        let mut out_col = vec![0.0; num_patches];
        let mut max_indices = vec![0usize; num_patches];

        for p in 0..num_patches {
            let base = p * patch_size;
            let mut max_val = f32::NEG_INFINITY;
            let mut max_idx = 0;
            for k in 0..patch_size {
                let idx = base + k;
                if idx < x_col.len() && x_col[idx] > max_val {
                    max_val = x_col[idx];
                    max_idx = k;
                }
            }
            out_col[p] = max_val;
            max_indices[p] = max_idx;
        }

        let mut out_data = vec![0.0; n * c * out_h * out_w];
        for ni in 0..n {
            for ci in 0..c {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let patch_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                        let src = ((ni * c + ci) * out_h * out_w) + (ni * c + ci) * out_h * out_w + oy * out_w + ox;
                        if src / (out_h * out_w) == (ni * c + ci) {
                            let offset = (ni * c + ci) * out_h * out_w;
                            out_data[(ni * c + ci) * out_h * out_w + oy * out_w + ox] = out_col[offset + oy * out_w + ox];
                        }
                    }
                }
            }
        }

        let shape = vec![n, c, out_h, out_w];
        Tensor::new(out_data, shape, x.requires_grad)
    }
}

impl AvgPool2d {
    pub fn new(kernel_size: usize, stride: Option<usize>) -> Self {
        AvgPool2d {
            kernel_size,
            stride: stride.unwrap_or(kernel_size),
        }
    }
}

impl Module for AvgPool2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];

        let out_h = (h - self.kernel_size) / self.stride + 1;
        let out_w = (w - self.kernel_size) / self.stride + 1;

        let mut x_reshaped = x.data.borrow().clone();
        let x_col = im2col_for_pool(&x_reshaped, &[n * c, 1, h, w], self.kernel_size, self.stride);
        let patch_size = self.kernel_size * self.kernel_size;
        let num_patches = n * c * out_h * out_w;

        let mut out_col = vec![0.0; num_patches];

        for p in 0..num_patches {
            let base = p * patch_size;
            let mut sum = 0.0f32;
            for k in 0..patch_size {
                let idx = base + k;
                if idx < x_col.len() {
                    sum += x_col[idx];
                }
            }
            out_col[p] = sum / (patch_size as f32);
        }

        let mut out_data = vec![0.0; n * c * out_h * out_w];
        for ni in 0..n {
            for ci in 0..c {
                for oy in 0..out_h {
                    for ox in 0..out_w {
                        let patch_idx = ((ni * c + ci) * out_h + oy) * out_w + ox;
                        let src = ((ni * c + ci) * out_h * out_w) + (oy * out_w + ox);
                        if src < out_col.len() {
                            out_data[patch_idx] = out_col[src];
                        }
                    }
                }
            }
        }

        let shape = vec![n, c, out_h, out_w];
        Tensor::new(out_data, shape, x.requires_grad)
    }
}

impl Flatten {
    pub fn new() -> Self {
        Flatten
    }
}

impl Module for Flatten {
    fn forward(&self, x: &Tensor) -> Tensor {
        let batch_size = x.shape[0];
        let mut size = 1;
        for i in 1..x.shape.len() {
            size *= x.shape[i];
        }
        let data = x.data.borrow().clone();
        let shape = vec![batch_size, size];
        Tensor::new(data, shape, x.requires_grad)
    }
}

impl BatchNorm2d {
    pub fn new(num_channels: usize, eps: f32, momentum: f32) -> Self {
        BatchNorm2d {
            num_channels,
            eps,
            momentum,
            weight: Tensor::new(vec![1.0; num_channels], vec![num_channels], true),
            bias: Tensor::new(vec![0.0; num_channels], vec![num_channels], true),
            running_mean: vec![0.0; num_channels],
            running_var: vec![1.0; num_channels],
            training: true,
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];

        let (mean, var) = if self.training {
            let mut mean = vec![0.0; c];
            let mut var = vec![0.0; c];

            for ci in 0..c {
                let mut sum = 0.0f32;
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let idx = (ni * c + ci) * h * w + hi * w + wi;
                            if idx < x.data.borrow().len() {
                                sum += x.data.borrow()[idx];
                            }
                        }
                    }
                }
                mean[ci] = sum / ((n * h * w) as f32);
            }

            for ci in 0..c {
                let mut var_sum = 0.0f32;
                for ni in 0..n {
                    for hi in 0..h {
                        for wi in 0..w {
                            let idx = (ni * c + ci) * h * w + hi * w + wi;
                            if idx < x.data.borrow().len() {
                                let diff = x.data.borrow()[idx] - mean[ci];
                                var_sum += diff * diff;
                            }
                        }
                    }
                }
                var[ci] = var_sum / ((n * h * w) as f32);
            }

            (mean, var)
        } else {
            (self.running_mean.clone(), self.running_var.clone())
        };

        let mut out_data = vec![0.0; n * c * h * w];

        for ni in 0..n {
            for ci in 0..c {
                let inv = 1.0 / (var[ci].sqrt() + self.eps);
                let gamma = self.weight.data.borrow()[ci];
                let beta = self.bias.data.borrow()[ci];

                for hi in 0..h {
                    for wi in 0..w {
                        let src_idx = (ni * c + ci) * h * w + hi * w + wi;
                        let dst_idx = src_idx;
                        if src_idx < x.data.borrow().len() && dst_idx < out_data.len() {
                            let x_centered = x.data.borrow()[src_idx] - mean[ci];
                            let x_norm = x_centered * inv;
                            out_data[dst_idx] = gamma * x_norm + beta;
                        }
                    }
                }
            }
        }

        let shape = vec![n, c, h, w];
        Tensor::new(out_data, shape, x.requires_grad)
    }
}

impl Dropout2d {
    pub fn new(p: f32) -> Self {
        Dropout2d {
            p,
            training: true,
        }
    }

    pub fn eval(&mut self) {
        self.training = false;
    }

    pub fn train(&mut self) {
        self.training = true;
    }
}

impl Module for Dropout2d {
    fn forward(&self, x: &Tensor) -> Tensor {
        if !self.training || self.p == 0.0 {
            return Tensor::new(x.data.borrow().clone(), x.shape.clone(), x.requires_grad);
        }

        let n = x.shape[0];
        let c = x.shape[1];
        let h = x.shape[2];
        let w = x.shape[3];

        let mut mask = vec![0.0; n * c * h * w];
        let scale = 1.0 / (1.0 - self.p);

        for i in 0..mask.len() {
            let r: f32 = rand::random();
            mask[i] = if r > self.p { scale } else { 0.0 };
        }

        let data = x.data.borrow();
        let out_data: Vec<f32> = data.iter().zip(mask.iter()).map(|(x, m)| x * m).collect();

        Tensor::new(out_data, x.shape.clone(), x.requires_grad)
    }
}