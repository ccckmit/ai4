import { Tensor } from './tensor';

export abstract class Module {
  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const [key, val] of Object.entries(this)) {
      if (key.startsWith('_')) continue;
      if (val instanceof Tensor && val.requires_grad) params.push(val);
      else if (val instanceof Module) params.push(...val.parameters());
      else if (Array.isArray(val)) {
        for (const item of val) {
          if (item instanceof Module) params.push(...item.parameters());
        }
      }
    }
    return params;
  }

  abstract forward(x: Tensor): Tensor;
  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class Linear extends Module {
  weight: Tensor;
  bias: Tensor | null;

  constructor(in_features: number, out_features: number, bias = false) {
    super();
    const std = 0.08;
    const w: number[] = [];
    for (let i = 0; i < out_features; i++)
      for (let j = 0; j < in_features; j++)
        w.push((Math.random() * 2 - 1) * std);
    this.weight = new Tensor(w, [out_features, in_features], true);
    this.bias = bias ? new Tensor(new Array(out_features).fill(0), [out_features], true) : null;
  }

  forward(x: Tensor): Tensor {
    const out = x.matmul(this.weight);
    return this.bias ? out.add(this.bias) : out;
  }
}

export class ReLU extends Module {
  forward(x: Tensor): Tensor { return x.relu(); }
  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class Tanh extends Module {
  forward(x: Tensor): Tensor { return x.tanh(); }
  __call__(x: Tensor): Tensor { return this.forward(x); }
}

export class Embedding extends Module {
  weight: Tensor;

  constructor(num_embeddings: number, embedding_dim: number) {
    super();
    const w: number[] = [];
    for (let i = 0; i < num_embeddings; i++)
      for (let j = 0; j < embedding_dim; j++)
        w.push((Math.random() * 2 - 1) * 0.08);
    this.weight = new Tensor(w, [num_embeddings, embedding_dim], true);
  }

  forward(indices: Tensor): Tensor {
    const [batch, seq_len] = indices.shape;
    const vocabSize = this.weight.shape[1];
    const out: number[] = new Array(batch * seq_len * vocabSize);

    for (let b = 0; b < batch; b++) {
      for (let t = 0; t < seq_len; t++) {
        const idx = Math.round(indices.data[b * seq_len + t]);
        const embIdx = Math.max(0, Math.min(idx, this.weight.shape[0] - 1));
        for (let j = 0; j < vocabSize; j++) {
          out[(b * seq_len + t) * vocabSize + j] = this.weight.data[embIdx * vocabSize + j];
        }
      }
    }

    const resultShape = [batch, seq_len, vocabSize];
    const result = new Tensor(out, resultShape, this.weight.requires_grad);
    result._prev = [this.weight];

    result._backward = () => {
      if (!this.weight.requires_grad) return;
      for (let b = 0; b < batch; b++) {
        for (let t = 0; t < seq_len; t++) {
          const idx = Math.round(indices.data[b * seq_len + t]);
          const embIdx = Math.max(0, Math.min(idx, this.weight.shape[0] - 1));
          for (let j = 0; j < vocabSize; j++) {
            this.weight.grad[embIdx * vocabSize + j] += result.grad[(b * seq_len + t) * vocabSize + j] ?? 0;
          }
        }
      }
    };
    return result;
  }
}

export class RMSNorm extends Module {
  eps = 1e-5;
  scale: Tensor;

  constructor(dim: number) {
    super();
    this.scale = new Tensor(new Array(dim).fill(1), [dim], false);
  }

  forward(x: Tensor): Tensor {
    const ms = x.data.map((_, rowIdx) => {
      let m2 = 0;
      const rowStart = rowIdx * x.shape[x.shape.length - 1];
      for (let j = 0; j < x.shape[x.shape.length - 1]; j++) {
        const v = x.data[rowStart + j];
        m2 += v * v;
      }
      const d = x.shape[x.shape.length - 1];
      return Math.sqrt(m2 / d + this.eps);
    });
    const out: number[] = [];
    for (let i = 0; i < x.data.length; i++) {
      const row = Math.floor(i / x.shape[x.shape.length - 1]);
      out.push(x.data[i] / ms[row]);
    }
    const result = new Tensor(out, x.shape, x.requires_grad);
    result._prev = [x];
    result._backward = () => {
      if (!x.requires_grad) return;
      for (let i = 0; i < x.data.length; i++) {
        const row = Math.floor(i / x.shape[x.shape.length - 1]);
        x.grad[i] += result.grad[i] / ms[row];
      }
    };
    return result;
  }
}

export function mse_loss(pred: Tensor, target: Tensor): Tensor {
  const diff = pred.sub(target);
  const sq = diff.mul(diff);
  return sq.mean();
}

export class Adam {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  m: number[][] = [];
  v: number[][] = [];
  t = 0;

  constructor(params: Tensor[], lr = 0.01, betas: [number, number] = [0.9, 0.999], eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.m = params.map(p => p.data.map(() => 0));
    this.v = params.map(p => p.data.map(() => 0));
  }

  step(): void {
    this.t++;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      for (let j = 0; j < p.data.length; j++) {
        const g = p.grad[j];
        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * g;
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * g * g;
        const m_hat = this.m[i][j] / (1 - this.beta1 ** this.t);
        const v_hat = this.v[i][j] / (1 - this.beta2 ** this.t);
        p.data[j] -= this.lr * m_hat / (Math.sqrt(v_hat) + this.eps);
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}

export class Sequential extends Module {
  layers: Module[];

  constructor(layers: Module[]) {
    super();
    this.layers = layers;
  }

  forward(x: Tensor): Tensor {
    for (const layer of this.layers) x = layer.forward(x);
    return x;
  }
}