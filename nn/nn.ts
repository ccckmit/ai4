import { Tensor } from './tensor';

export abstract class Module {
  parameters(): Tensor[] {
    const params: Tensor[] = [];
    for (const v of Object.values(this)) {
      if (v instanceof Tensor && v.requires_grad) params.push(v);
      else if (v instanceof Module) params.push(...v.parameters());
      else if (Array.isArray(v) && v.some(item => item instanceof Module)) {
        for (const item of v) if (item instanceof Module) params.push(...item.parameters());
      }
    }
    return params;
  }
}

export class Linear extends Module {
  weight: Tensor;
  bias: Tensor | null;

  constructor(in_features: number, out_features: number, bias = false) {
    super();
    const std = 0.08;
    this.weight = Tensor.from(
      Array(out_features).fill(0).map(() =>
        Array(in_features).fill(0).map(() => (Math.random() * 2 - 1) * std)
      ),
      true
    );
    this.bias = bias
      ? Tensor.from([Array(out_features).fill(0)], true)
      : null;
  }

  forward(x: Tensor): Tensor {
    const out = x.matmul(this.weight.transpose());
    if (this.bias) return out.add(this.bias);
    return out;
  }

  __call__(x: Tensor): Tensor {
    return this.forward(x);
  }
}

export class Embedding extends Module {
  weight: Tensor;

  constructor(num_embeddings: number, embedding_dim: number) {
    super();
    this.weight = Tensor.from(
      Array(num_embeddings).fill(0).map(() =>
        Array(embedding_dim).fill(0).map(() => (Math.random() * 2 - 1) * 0.08)
      ),
      true
    );
  }

  forward(indices: Tensor): Tensor {
    const idx = indices.data.map(row => row.map(v => Math.round(Number(v))));
    const out: number[][] = idx.flatMap(row =>
      row.map(i => this.weight.data[i] ?? this.weight.data[0] ?? [])
    );
    const result = new Tensor(out, [this.weight]);
    (result as any)._backward = () => {
      const grad = result.grad;
      for (const row of idx) {
        for (const i of row) {
          for (let j = 0; j < (this.weight.grad[i]?.length ?? 0); j++) {
            this.weight.grad[i][j] += grad[idx.indexOf(row) as number]?.[j] ?? 0;
          }
        }
      }
    };
    return result;
  }

  __call__(indices: Tensor): Tensor {
    return this.forward(indices);
  }
}

export class RMSNorm extends Module {
  eps = 1e-5;
  scale: Tensor;

  constructor(dim: number) {
    super();
    this.scale = Tensor.from([Array(dim).fill(1)], false);
  }

  forward(x: Tensor): Tensor {
    const ms = x.data.map(row => {
      const m2 = row.reduce((s, v) => s + v * v, 0) / row.length;
      return Math.sqrt(m2 + this.eps);
    });
    const out = x.data.map((row, i) => row.map(v => v / ms[i]));
    const result = new Tensor(out, [x]);
    result.requires_grad = x.requires_grad;
    result._backward = () => {
      if (!x.requires_grad) return;
      const inv_std = ms.map(m => 1 / m);
      for (let i = 0; i < x.grad.length; i++) {
        for (let j = 0; j < x.grad[i].length; j++) {
          x.grad[i][j] += result.grad[i]?.[j] ?? 0 * inv_std[i];
        }
      }
    };
    return result;
  }

  __call__(x: Tensor): Tensor {
    return this.forward(x);
  }
}

export class Adam {
  params: Tensor[];
  lr: number;
  beta1: number;
  beta2: number;
  eps: number;
  m: number[][][] = [];
  v: number[][][] = [];
  t = 0;

  constructor(params: Tensor[], lr = 0.01, betas: [number, number] = [0.85, 0.99], eps = 1e-8) {
    this.params = params;
    this.lr = lr;
    this.beta1 = betas[0];
    this.beta2 = betas[1];
    this.eps = eps;
    this.m = params.map(p => p.data.map(row => row.map(() => 0)));
    this.v = params.map(p => p.data.map(row => row.map(() => 0)));
  }

  step(): void {
    this.t++;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      for (let j = 0; j < p.data.length; j++) {
        for (let k = 0; k < p.data[j].length; k++) {
          const g = p.grad[j]?.[k] ?? 0;
          this.m[i][j][k] = this.beta1 * this.m[i][j][k] + (1 - this.beta1) * g;
          this.v[i][j][k] = this.beta2 * this.v[i][j][k] + (1 - this.beta2) * g * g;
          const m_hat = this.m[i][j][k] / (1 - this.beta1 ** this.t);
          const v_hat = this.v[i][j][k] / (1 - this.beta2 ** this.t);
          p.data[j][k] -= this.lr * m_hat / (Math.sqrt(v_hat) + this.eps);
        }
      }
    }
  }

  zeroGrad(): void {
    for (const p of this.params) p.zeroGrad();
  }
}