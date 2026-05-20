/**
 * Tensor with automatic differentiation (autograd) based on plain JS arrays.
 * Records operation history to support backpropagation through a computation graph.
 */

function unbroadcast(grad: number[][], shape: number[]): number[][] {
  const gradShape = [grad.length, grad[0]?.length ?? 0];
  if (gradShape[0] === shape[0] && gradShape[1] === shape[1]) return grad;
  const result: number[][] = [];
  for (let i = 0; i < shape[0]; i++) {
    result.push(new Array(shape[1]).fill(0));
  }
  for (let i = 0; i < grad.length; i++) {
    for (let j = 0; j < grad[i].length; j++) {
      result[i % shape[0]][j % shape[1]] += grad[i][j];
    }
  }
  return result;
}

export class Tensor {
  data: number[][];
  grad: number[][] = [];
  requires_grad: boolean;
  _backward: () => void = () => {};
  _prev: Set<Tensor> = new Set();

  constructor(
    data: number[][],
    children: Tensor[] = [],
    requires_grad = false
  ) {
    this.data = data;
    this._prev = new Set(children);
    this.requires_grad = requires_grad;
    if (requires_grad) {
      this.grad = this.data.map(row => new Array(row.length).fill(0));
    }
  }

  get shape(): number[] {
    return [this.data.length, this.data[0]?.length ?? 0];
  }

  zeroGrad(): void {
    this.grad = this.data.map(row => new Array(row.length).fill(0));
  }

  backward(): void {
    const topo: Tensor[] = [];
    const visited = new Set<Tensor>();

    const build = (v: Tensor) => {
      if (visited.has(v)) return;
      visited.add(v);
      for (const child of v._prev) build(child);
      topo.push(v);
    };
    build(this);

    this.grad = this.data.map(row => new Array(row.length).fill(1));
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  // Operators
  add(other: Tensor | number[][] | number): Tensor {
    const b = other instanceof Tensor ? other : new Tensor(Array.isArray(other) ? other : [[other]]);
    const shape = [Math.max(this.data.length, b.data.length), Math.max(this.data[0]?.length ?? 0, b.data[0]?.length ?? 0)];
    const out = Tensor.broadcast(this, b, shape);
    out._prev = new Set([this, b]);
    out.requires_grad = this.requires_grad || b.requires_grad;
    out._backward = () => {
      if (this.requires_grad) {
        const g = unbroadcast(out.grad, [this.data.length, this.data[0]?.length ?? 0]);
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            this.grad[i][j] += g[i]?.[j] ?? 0;
      }
      if (b.requires_grad) {
        const g = unbroadcast(out.grad, [b.data.length, b.data[0]?.length ?? 0]);
        for (let i = 0; i < b.grad.length; i++)
          for (let j = 0; j < b.grad[i].length; j++)
            b.grad[i][j] += g[i]?.[j] ?? 0;
      }
    };
    return out;
  }

  mul(other: Tensor | number[][] | number): Tensor {
    const b = other instanceof Tensor ? other : new Tensor(Array.isArray(other) ? other : [[other]]);
    const out = new Tensor(this.data.map((row, i) => row.map((x, j) => x * (b.data[i]?.[j] ?? 0))), [this, b]);
    out.requires_grad = this.requires_grad || b.requires_grad;
    out._backward = () => {
      if (this.requires_grad) {
        const g = out.grad.map((row, i) => row.map((x, j) => x * (b.data[i]?.[j] ?? 0)));
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            this.grad[i][j] += g[i]?.[j] ?? 0;
      }
      if (b.requires_grad) {
        const g = out.grad.map((row, i) => row.map((x, j) => x * (this.data[i]?.[j] ?? 0)));
        for (let i = 0; i < b.grad.length; i++)
          for (let j = 0; j < b.grad[i].length; j++)
            b.grad[i][j] += g[i]?.[j] ?? 0;
      }
    };
    return out;
  }

  matmul(other: Tensor): Tensor {
    const a = this.data, b = other.data;
    const out = a.map(rowA => b[0].map((_, j) =>
      rowA.reduce((sum, aik, k) => sum + aik * (b[k]?.[j] ?? 0), 0)
    ));
    const result = new Tensor(out, [this, other]);
    result.requires_grad = this.requires_grad || other.requires_grad;
    result._backward = () => {
      if (this.requires_grad) {
        const ga = new Array(this.data.length).fill(0).map(() => new Array(this.data[0].length).fill(0));
        for (let i = 0; i < ga.length; i++)
          for (let j = 0; j < ga[i].length; j++)
            for (let k = 0; k < other.data.length; k++)
              ga[i][j] += result.grad[i]?.[k] ?? 0 * (other.data[k]?.[j] ?? 0);
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            this.grad[i][j] += ga[i]?.[j] ?? 0;
      }
      if (other.requires_grad) {
        const gb = new Array(other.data.length).fill(0).map(() => new Array(other.data[0].length).fill(0));
        for (let i = 0; i < gb.length; i++)
          for (let j = 0; j < gb[i].length; j++)
            for (let k = 0; k < this.data.length; k++)
              gb[i][j] += result.grad[k]?.[i] ?? 0 * (this.data[k]?.[j] ?? 0);
        for (let i = 0; i < other.grad.length; i++)
          for (let j = 0; j < other.grad[i].length; j++)
            other.grad[i][j] += gb[i]?.[j] ?? 0;
      }
    };
    return result;
  }

  relu(): Tensor {
    const out = this.data.map(row => row.map(x => Math.max(0, x)));
    const result = new Tensor(out, [this]);
    result.requires_grad = this.requires_grad;
    result._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            if (this.data[i]?.[j] > 0)
              this.grad[i][j] += result.grad[i]?.[j] ?? 0;
      }
    };
    return result;
  }

  sum(): Tensor {
    let total = 0;
    for (const row of this.data)
      for (const v of row) total += v;
    const result = new Tensor([[total]], [this]);
    result.requires_grad = this.requires_grad;
    result._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            this.grad[i][j] += result.grad[0]?.[0] ?? 0;
      }
    };
    return result;
  }

  transpose(): Tensor {
    const out = this.data[0].map((_, j) => this.data.map(row => row[j]));
    const result = new Tensor(out, [this]);
    result.requires_grad = this.requires_grad;
    result._backward = () => {
      if (this.requires_grad && result.grad && result.grad[0]) {
        const g = result.grad[0].map((_: number, j: number) => result.grad!.map((row: number[]) => row[j] ?? 0));
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++)
            this.grad[i][j] += g[i]?.[j] ?? 0;
      }
    };
    return result;
  }

  softmax(): Tensor {
    const maxVal = Math.max(...this.data.flat());
    const exps = this.data.map(row => row.map(x => Math.exp(x - maxVal)));
    const sum = exps.flat().reduce((a, b) => a + b, 0);
    const probs = exps.map(row => row.map(x => x / sum));
    const result = new Tensor(probs, [this]);
    result.requires_grad = this.requires_grad;
    result._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++)
          for (let j = 0; j < this.grad[i].length; j++) {
            const s = probs[i]?.[j] ?? 0;
            this.grad[i][j] += result.grad[i]?.[j] ?? 0 * s * (1 - s);
          }
      }
    };
    return result;
  }

  neg(): Tensor { return this.mul(-1); }

  cross_entropy(targets: Tensor | number[][]): Tensor {
    const targetData = targets instanceof Tensor ? targets.data : targets;
    const batch_size = targetData.length;
    const seq_len = targetData[0]?.length ?? 1;
    const vocab_size = this.data[0]?.length ?? 1;

    const flat_logits: number[][] = this.data;
    const max_logits = flat_logits.map(row => Math.max(...row));
    const exps = flat_logits.map((row, bi) =>
      row.map(v => Math.exp(v - max_logits[bi]))
    );
    const sum_exps = exps.map(row => row.reduce((a, b) => a + b, 0));
    const probs = exps.map((row, bi) => row.map(v => v / (sum_exps[bi] + 1e-10)));

    let loss = 0;
    for (let b = 0; b < batch_size; b++) {
      for (let t = 0; t < seq_len; t++) {
        const idx = targetData[b]?.[t] ?? 0;
        loss -= Math.log(probs[b * seq_len + t]?.[idx] ?? 1e-10);
      }
    }
    loss /= batch_size * seq_len;

    const out = new Tensor([[loss]], [this]);
    out.requires_grad = this.requires_grad;
    out._backward = () => {
      if (!this.requires_grad) return;
      for (let b = 0; b < batch_size; b++) {
        for (let t = 0; t < seq_len; t++) {
          const idx = targetData[b]?.[t] ?? 0;
          const row_idx = b * seq_len + t;
          for (let v = 0; v < vocab_size; v++) {
            const prob = probs[row_idx]?.[v] ?? 0;
            const grad_val = (prob - (v === idx ? 1 : 0)) / (batch_size * seq_len);
            const row = this.grad[row_idx];
            if (row) row[v] = (row[v] ?? 0) + grad_val;
          }
        }
      }
    };
    return out;
  }

  sub(other: Tensor | number): Tensor {
    if (other instanceof Tensor) {
      const neg = other.mul(-1);
      return this.add(neg);
    }
    return this.add(-Number(other));
  }

  // Static helpers
  private static broadcast(a: Tensor, b: Tensor, shape: number[]): Tensor {
    const data = shape[0] === a.data.length ? a.data : a.data;
    const bdata = shape[0] === b.data.length ? b.data : b.data;
    return new Tensor(data, [a, b]);
  }

  // Make from nested arrays
  static from(data: number[][] | number[], requires_grad = false): Tensor {
    const arr = data.map(row => Array.isArray(row) ? row : [row]);
    return new Tensor(arr, [], requires_grad);
  }
}

export function cat(tensors: Tensor[], axis = 0): Tensor {
  const data = axis === 0
    ? tensors.flatMap(t => t.data)
    : tensors[0].data.map((_, j) => tensors.flatMap(t => t.data.map(row => row[j] ?? 0)));
  const result = new Tensor(data, tensors);
  result.requires_grad = tensors.some(t => t.requires_grad);
  result._backward = () => {
    if (!result.requires_grad) return;
    let idx = 0;
    for (const t of tensors) {
      if (t.requires_grad) {
        const size = axis === 0 ? t.data.length : t.data[0]?.length ?? 0;
        const gradChunk = axis === 0
          ? result.grad.slice(idx, idx + size)
          : result.grad.map(row => row.slice(idx, idx + size));
        for (let i = 0; i < t.grad.length; i++)
          for (let j = 0; j < t.grad[i].length; j++)
            t.grad[i][j] += gradChunk[i]?.[j] ?? 0;
      }
      idx += axis === 0 ? t.data.length : t.data[0]?.length ?? 0;
    }
  };
  return result;
}