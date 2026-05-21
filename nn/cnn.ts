import { Tensor } from './tensor';

/** im2col helper: extract sliding patches from a 4D tensor [N,C,H,W] */
function im2col(
  data: number[],
  N: number, C: number, H: number, W: number,
  kH: number, kW: number,
  stride: number,
  pad: number
): number[] {
  const H2 = H + 2 * pad;
  const W2 = W + 2 * pad;
  const outH = (H - kH + 2 * pad) / stride + 1;
  const outW = (W - kW + 2 * pad) / stride + 1;

  const padded: number[] = new Array(N * C * H2 * W2).fill(0);
  for (const n of new Array(N).keys()) {
    for (const c of new Array(C).keys()) {
      for (const h of new Array(H).keys()) {
        for (const w of new Array(W).keys()) {
          const src = n * C * H * W + c * H * W + h * W + w;
          const dst = n * C * H2 * W2 + c * H2 * W2 + (h + pad) * W2 + (w + pad);
          padded[dst] = data[src];
        }
      }
    }
  }

  const col: number[] = [];
  for (let c = 0; c < C; c++) {
    for (let kh = 0; kh < kH; kh++) {
      for (let kw = 0; kw < kW; kw++) {
        for (let n = 0; n < N; n++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const h = kh + oh * stride;
              const w = kw + ow * stride;
              const idx = n * C * H2 * W2 + c * H2 * W2 + h * W2 + w;
              col.push(padded[idx]);
            }
          }
        }
      }
    }
  }
  return col;
}

/** col2im: distribute gradient back to original positions */
function col2im(
  col: number[],
  N: number, C: number, H: number, W: number,
  kH: number, kW: number,
  stride: number,
  pad: number
): number[] {
  const H2 = H + 2 * pad;
  const W2 = W + 2 * pad;
  const outH = (H - kH + 2 * pad) / stride + 1;
  const outW = (W - kW + 2 * pad) / stride + 1;

  const padded: number[] = new Array(N * C * H2 * W2).fill(0);
  let colIdx = 0;
  for (let c = 0; c < C; c++) {
    for (let kh = 0; kh < kH; kh++) {
      for (let kw = 0; kw < kW; kw++) {
        for (let n = 0; n < N; n++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const h = kh + oh * stride;
              const w = kw + ow * stride;
              const idx = n * C * H2 * W2 + c * H2 * W2 + h * W2 + w;
              padded[idx] += col[colIdx++];
            }
          }
        }
      }
    }
  }

  const out: number[] = new Array(N * C * H * W).fill(0);
  for (let n = 0; n < N; n++) {
    for (let c = 0; c < C; c++) {
      for (let h = 0; h < H; h++) {
        for (let w = 0; w < W; w++) {
          out[n * C * H * W + c * H * W + h * W + w] = padded[n * C * H2 * W2 + c * H2 * W2 + (h + pad) * W2 + (w + pad)];
        }
      }
    }
  }
  return out;
}

export class Conv2d {
  in_channels: number;
  out_channels: number;
  kernel_size: number;
  stride: number;
  padding: number;
  weight: Tensor;
  bias: Tensor | null;

  constructor(
    in_channels: number,
    out_channels: number,
    kernel_size: number,
    stride = 1,
    padding = 0,
    bias = true
  ) {
    this.in_channels = in_channels;
    this.out_channels = out_channels;
    this.kernel_size = kernel_size;
    this.stride = stride;
    this.padding = padding;

    const scale = Math.sqrt(2.0 / (in_channels * kernel_size * kernel_size));
    const w: number[] = [];
    for (let oc = 0; oc < out_channels; oc++)
      for (let ic = 0; ic < in_channels; ic++)
        for (let kh = 0; kh < kernel_size; kh++)
          for (let kw = 0; kw < kernel_size; kw++)
            w.push((Math.random() * 2 - 1) * scale);

    this.weight = new Tensor(w, [out_channels, in_channels, kernel_size, kernel_size], true);
    this.bias = bias
      ? new Tensor(new Array(out_channels).fill(0), [out_channels], true)
      : null;
  }

  forward(x: Tensor): Tensor {
    const [N, C, H, W] = x.shape;
    const outH = Math.floor((H + 2 * this.padding - this.kernel_size) / this.stride + 1);
    const outW = Math.floor((W + 2 * this.padding - this.kernel_size) / this.stride + 1);

    const xCol = im2col(x.data, N, C, H, W, this.kernel_size, this.kernel_size, this.stride, this.padding);
    const wRow = this.weight.data; // [OC, IC*KH*KW]
    const kPerOC = this.in_channels * this.kernel_size * this.kernel_size;
    const OH_OW = outH * outW;
    const N_outH_outW = N * OH_OW;
    const outData: number[] = new Array(N * this.out_channels * OH_OW).fill(0);

    for (let oc = 0; oc < this.out_channels; oc++) {
      const oc_offset = oc * kPerOC;
      for (let colRow = 0; colRow < kPerOC; colRow++) {
        const wVal = wRow[oc_offset + colRow];
        const colRow_offset = colRow * N_outH_outW;
        for (let n = 0; n < N; n++) {
          const n_col_offset = n * OH_OW;
          const n_out_offset = (n * this.out_channels + oc) * OH_OW;
          for (let spatial = 0; spatial < OH_OW; spatial++) {
            outData[n_out_offset + spatial] += wVal * xCol[colRow_offset + n_col_offset + spatial];
          }
        }
      }
      if (this.bias) {
        const bVal = this.bias.data[oc];
        for (let n = 0; n < N; n++) {
          const n_out_offset = (n * this.out_channels + oc) * OH_OW;
          for (let spatial = 0; spatial < OH_OW; spatial++) {
            outData[n_out_offset + spatial] += bVal;
          }
        }
      }
    }

    const result = new Tensor(outData, [N, this.out_channels, outH, outW], x.requires_grad || this.weight.requires_grad);
    result._prev = [x, this.weight];
    if (this.bias) result._prev.push(this.bias);

    result._backward = () => {
      if (x.requires_grad) {
        const dout = result.grad;
        const dcol: number[] = new Array(kPerOC * N_outH_outW).fill(0);
        for (let oc = 0; oc < this.out_channels; oc++) {
          const oc_offset = oc * kPerOC;
          for (let colRow = 0; colRow < kPerOC; colRow++) {
            const wVal = wRow[oc_offset + colRow];
            const colRow_offset = colRow * N_outH_outW;
            for (let n = 0; n < N; n++) {
              const n_col_offset = n * OH_OW;
              const n_out_offset = (n * this.out_channels + oc) * OH_OW;
              for (let spatial = 0; spatial < OH_OW; spatial++) {
                dcol[colRow_offset + n_col_offset + spatial] += wVal * dout[n_out_offset + spatial];
              }
            }
          }
        }
        const dxData = col2im(dcol, N, C, H, W, this.kernel_size, this.kernel_size, this.stride, this.padding);
        for (let i = 0; i < x.grad.length; i++) x.grad[i] += dxData[i];
      }
      if (this.weight.requires_grad) {
        const dout = result.grad;
        for (let oc = 0; oc < this.out_channels; oc++) {
          const oc_offset = oc * kPerOC;
          for (let colRow = 0; colRow < kPerOC; colRow++) {
            let sum = 0;
            const colRow_offset = colRow * N_outH_outW;
            for (let n = 0; n < N; n++) {
              const n_col_offset = n * OH_OW;
              const n_out_offset = (n * this.out_channels + oc) * OH_OW;
              for (let spatial = 0; spatial < OH_OW; spatial++) {
                sum += dout[n_out_offset + spatial] * xCol[colRow_offset + n_col_offset + spatial];
              }
            }
            this.weight.grad[oc_offset + colRow] += sum;
          }
        }
      }
      if (this.bias?.requires_grad) {
        const dout = result.grad;
        for (let oc = 0; oc < this.out_channels; oc++) {
          let sum = 0;
          for (let n = 0; n < N; n++) {
            const n_out_offset = (n * this.out_channels + oc) * OH_OW;
            for (let spatial = 0; spatial < OH_OW; spatial++) {
              sum += dout[n_out_offset + spatial];
            }
          }
          this.bias!.grad[oc] += sum;
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class MaxPool2d {
  kernel_size: number;
  stride: number;

  constructor(kernel_size: number, stride?: number) {
    this.kernel_size = kernel_size;
    this.stride = stride ?? kernel_size;
  }

  forward(x: Tensor): Tensor {
    const [N, C, H, W] = x.shape;
    const outH = Math.floor((H - this.kernel_size) / this.stride + 1);
    const outW = Math.floor((W - this.kernel_size) / this.stride + 1);
    const out: number[] = new Array(N * C * outH * outW).fill(0);

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let maxVal = -Infinity;
            for (let kh = 0; kh < this.kernel_size; kh++) {
              for (let kw = 0; kw < this.kernel_size; kw++) {
                const h = oh * this.stride + kh;
                const w = ow * this.stride + kw;
                if (h < H && w < W) {
                  const val = x.data[n * C * H * W + c * H * W + h * W + w];
                  if (val > maxVal) maxVal = val;
                }
              }
            }
            out[n * C * outH * outW + c * outH * outW + oh * outW + ow] = maxVal;
          }
        }
      }
    }

    const result = new Tensor(out, [N, C, outH, outW], x.requires_grad);
    result._prev = [x];
    result._backward = () => {
      if (!x.requires_grad) return;
      const grad = result.grad;
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              let maxVal = -Infinity, maxH = 0, maxW = 0;
              for (let kh = 0; kh < this.kernel_size; kh++) {
                for (let kw = 0; kw < this.kernel_size; kw++) {
                  const h = oh * this.stride + kh;
                  const w = ow * this.stride + kw;
                  if (h < H && w < W) {
                    const val = x.data[n * C * H * W + c * H * W + h * W + w];
                    if (val > maxVal) { maxVal = val; maxH = h; maxW = w; }
                  }
                }
              }
              const gradIdx = n * C * outH * outW + c * outH * outW + oh * outW + ow;
              x.grad[n * C * H * W + c * H * W + maxH * W + maxW] += grad[gradIdx];
            }
          }
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class AvgPool2d {
  kernel_size: number;
  stride: number;

  constructor(kernel_size: number, stride?: number) {
    this.kernel_size = kernel_size;
    this.stride = stride ?? kernel_size;
  }

  forward(x: Tensor): Tensor {
    const [N, C, H, W] = x.shape;
    const outH = Math.floor((H - this.kernel_size) / this.stride + 1);
    const outW = Math.floor((W - this.kernel_size) / this.stride + 1);
    const kArea = this.kernel_size * this.kernel_size;
    const out: number[] = new Array(N * C * outH * outW).fill(0);

    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let oh = 0; oh < outH; oh++) {
          for (let ow = 0; ow < outW; ow++) {
            let sum = 0;
            for (let kh = 0; kh < this.kernel_size; kh++) {
              for (let kw = 0; kw < this.kernel_size; kw++) {
                const h = oh * this.stride + kh;
                const w = ow * this.stride + kw;
                if (h < H && w < W) sum += x.data[n * C * H * W + c * H * W + h * W + w];
              }
            }
            out[n * C * outH * outW + c * outH * outW + oh * outW + ow] = sum / kArea;
          }
        }
      }
    }

    const result = new Tensor(out, [N, C, outH, outW], x.requires_grad);
    result._prev = [x];
    const ks = this.kernel_size;
    result._backward = () => {
      if (!x.requires_grad) return;
      const grad = result.grad;
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let oh = 0; oh < outH; oh++) {
            for (let ow = 0; ow < outW; ow++) {
              const g = (grad[n * C * outH * outW + c * outH * outW + oh * outW + ow] ?? 0) / kArea;
              for (let kh = 0; kh < ks; kh++) {
                for (let kw = 0; kw < ks; kw++) {
                  const h = oh * this.stride + kh;
                  const w = ow * this.stride + kw;
                  if (h < H && w < W) {
                    x.grad[n * C * H * W + c * H * W + h * W + w] += g;
                  }
                }
              }
            }
          }
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class Flatten {
  forward(x: Tensor): Tensor {
    const [N, ...rest] = x.shape;
    const flatDim = rest.reduce((a, b) => a * b, 1);
    const newShape = [N, flatDim];
    const result = new Tensor([...x.data], newShape, x.requires_grad);
    result._prev = [x];
    result._backward = () => {
      if (x.requires_grad) {
        for (let i = 0; i < x.grad.length; i++) x.grad[i] += result.grad[i];
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class BatchNorm2d {
  num_channels: number;
  eps: number;
  momentum: number;
  weight: Tensor;
  bias: Tensor;
  running_mean: number[];
  running_var: number[];
  training = true;

  constructor(num_channels: number, eps = 1e-5, momentum = 0.1) {
    this.num_channels = num_channels;
    this.eps = eps;
    this.momentum = momentum;
    this.weight = new Tensor(new Array(num_channels).fill(1), [num_channels], true);
    this.bias = new Tensor(new Array(num_channels).fill(0), [num_channels], true);
    this.running_mean = new Array(num_channels).fill(0);
    this.running_var = new Array(num_channels).fill(1);
  }

  forward(x: Tensor): Tensor {
    const [N, C, H, W] = x.shape;
    const result = new Tensor(new Array(x.data.length), x.shape, x.requires_grad || this.weight.requires_grad);

    if (this.training) {
      for (let c = 0; c < C; c++) {
        let mean = 0;
        for (let n = 0; n < N; n++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              mean += x.data[n * C * H * W + c * H * W + h * W + w];
            }
          }
        }
        mean /= N * H * W;
        this.running_mean[c] = (1 - this.momentum) * this.running_mean[c] + this.momentum * mean;

        let variance = 0;
        for (let n = 0; n < N; n++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              const diff = x.data[n * C * H * W + c * H * W + h * W + w] - mean;
              variance += diff * diff;
            }
          }
        }
        variance /= N * H * W;
        this.running_var[c] = (1 - this.momentum) * this.running_var[c] + this.momentum * variance;

        for (let n = 0; n < N; n++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              const idx = n * C * H * W + c * H * W + h * W + w;
              const std_inv = 1 / Math.sqrt(variance + this.eps);
              result.data[idx] = this.weight.data[c] * (x.data[idx] - mean) * std_inv + this.bias.data[c];
            }
          }
        }
      }
    } else {
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              const idx = n * C * H * W + c * H * W + h * W + w;
              const std_inv = 1 / Math.sqrt(this.running_var[c] + this.eps);
              result.data[idx] = this.weight.data[c] * (x.data[idx] - this.running_mean[c]) * std_inv + this.bias.data[c];
            }
          }
        }
      }
    }

    result._prev = [x, this.weight, this.bias];
    result._backward = () => {
      if (!x.requires_grad) return;
      const m = N * H * W;
      for (let c = 0; c < C; c++) {
        let gradGamma = 0, gradBeta = 0;
        for (let n = 0; n < N; n++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              const idx = n * C * H * W + c * H * W + h * W + w;
              gradGamma += result.grad[idx] * (x.data[idx] - (this.training ? this.running_mean[c] : this.running_mean[c])) * 1 / Math.sqrt((this.training ? this.running_var[c] : this.running_var[c]) + this.eps);
              gradBeta += result.grad[idx];
            }
          }
        }
        this.weight.grad[c] += gradGamma;
        this.bias.grad[c] += gradBeta;
      }
      for (let n = 0; n < N; n++) {
        for (let c = 0; c < C; c++) {
          for (let h = 0; h < H; h++) {
            for (let w = 0; w < W; w++) {
              const idx = n * C * H * W + c * H * W + h * W + w;
              const bn_var = this.training ? this.running_var[c] : this.running_var[c];
              const bn_mean = this.training ? this.running_mean[c] : this.running_mean[c];
              const std_inv = 1 / Math.sqrt(bn_var + this.eps);
              x.grad[idx] += result.grad[idx] * this.weight.data[c] * std_inv;
            }
          }
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
  eval() { this.training = false; }
  train() { this.training = true; }
}

export class Dropout2d {
  p: number;
  training = true;

  constructor(p = 0.5) {
    this.p = p;
  }

  forward(x: Tensor): Tensor {
    if (!this.training || this.p === 0) return x;
    const [N, C, H, W] = x.shape;
    const mask: number[] = [];
    for (let n = 0; n < N; n++) {
      for (let c = 0; c < C; c++) {
        for (let h = 0; h < H; h++) {
          for (let w = 0; w < W; w++) {
            mask.push(Math.random() < 1 - this.p ? 1 : 0);
          }
        }
      }
    }
    const out = x.data.map((v, i) => v * mask[i] / (1 - this.p));
    const result = new Tensor(out, x.shape, x.requires_grad);
    result._prev = [x];
    result._backward = () => {
      if (x.requires_grad) {
        for (let i = 0; i < x.grad.length; i++) {
          x.grad[i] += result.grad[i] * mask[i] / (1 - this.p);
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor { return this.forward(x); }
  eval() { this.training = false; }
  train() { this.training = true; }
}