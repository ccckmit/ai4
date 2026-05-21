/**
 * Tensor with N-dimensional automatic differentiation (autograd).
 * Data is stored as flat array + shape, matching tensor.rs design.
 */

export class Tensor {
  data: number[];
  grad: number[];
  shape: number[];
  requires_grad: boolean;
  _backward: () => void;
  _prev: Tensor[];

  constructor(data: number[], shape: number[], requires_grad = false) {
    this.data = data;
    this.shape = shape;
    this.grad = new Array(data.length).fill(0);
    this.requires_grad = requires_grad;
    this._backward = () => {};
    this._prev = [];
  }

  size(): number {
    return this.data.length;
  }

  strides(): number[] {
    const n = this.shape.length;
    const strides = new Array(n).fill(0);
    if (n === 0) return strides;
    strides[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
      strides[i] = strides[i + 1] * this.shape[i + 1];
    }
    return strides;
  }

  index(indices: number[]): number {
    const strides = this.strides();
    let idx = 0;
    for (let i = 0; i < indices.length; i++) {
      idx += indices[i] * strides[i];
    }
    return idx;
  }

  at(...indices: number[]): number {
    return this.data[this.index(indices)];
  }

  zeroGrad(): void {
    this.grad.fill(0);
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

    this.grad = new Array(this.data.length).fill(1);
    for (let i = topo.length - 1; i >= 0; i--) {
      topo[i]._backward();
    }
  }

  // --- Factory methods ---

  static zeros(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Array(size).fill(0), shape);
  }

  static ones(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    return new Tensor(new Array(size).fill(1), shape);
  }

  static randn(shape: number[]): Tensor {
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Array(size);
    for (let i = 0; i < size; i++) {
      let u = 0, v = 0;
      while (u === 0) u = Math.random();
      while (v === 0) v = Math.random();
      data[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
    }
    return new Tensor(data, shape);
  }

  static from(data: number[] | number[][] | number[][][] | number[][][][], requires_grad = false): Tensor {
    function flatten(arr: any): number[] {
      if (typeof arr[0] === 'number') return Array.from(arr);
      return arr.flatMap(flatten);
    }
    function shapeOf(arr: any): number[] {
      if (typeof arr[0] === 'number') return [arr.length];
      return [arr.length, ...shapeOf(arr[0])];
    }
    const flat = flatten(Array.isArray(data) ? data : [data]);
    const shp = shapeOf(Array.isArray(data) ? data : [data]);
    return new Tensor(flat, shp, requires_grad);
  }

  // --- Shape operations ---

  reshape(...shape: number[]): Tensor {
    const size = this.data.length;
    let known = 1, unknown = -1;
    for (const d of shape) {
      if (d === -1) unknown = unknown === -1 ? d : 0;
      else known *= d;
    }
    if (unknown !== -1) shape = shape.map(d => d === -1 ? size / known : d);
    const out = new Tensor([...this.data], shape, this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        this.grad = this.grad.map((v, i) => v + out.grad[i]);
      }
    };
    return out;
  }

  transpose(ax1 = 0, ax2 = 1): Tensor {
    const n = this.shape.length;
    const newShape = [...this.shape];
    [newShape[ax1], newShape[ax2]] = [newShape[ax2], newShape[ax1]];

    const strides = this.strides();
    const newStrides: number[] = [];
    newStrides[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) {
      newStrides[i] = newStrides[i + 1] * newShape[i + 1];
    }

    const outData: number[] = new Array(this.data.length);
    for (let flat = 0; flat < this.data.length; flat++) {
      const idx: number[] = [];
      let rem = flat;
      for (let d = 0; d < n; d++) {
        idx.push(Math.floor(rem / strides[d]));
        rem = rem % strides[d];
      }

      const newIdx = [...idx];
      [newIdx[ax1], newIdx[ax2]] = [newIdx[ax2], newIdx[ax1]];

      let newFlat = 0;
      for (let d = 0; d < n; d++) {
        newFlat += newIdx[d] * newStrides[d];
      }
      outData[newFlat] = this.data[flat];
    }

    const result = new Tensor(outData, newShape, this.requires_grad);
    result._prev = [this];
    result._backward = () => {
      if (this.requires_grad) {
        for (let flat = 0; flat < this.data.length; flat++) {
          const idx: number[] = [];
          let rem = flat;
          for (let d = 0; d < n; d++) {
            idx.push(Math.floor(rem / strides[d]));
            rem = rem % strides[d];
          }
          const newIdx = [...idx];
          [newIdx[ax1], newIdx[ax2]] = [newIdx[ax2], newIdx[ax1]];
          let newFlat = 0;
          for (let d = 0; d < n; d++) {
            newFlat += newIdx[d] * newStrides[d];
          }
          this.grad[flat] += result.grad[newFlat];
        }
      }
    };
    return result;
  }

  // --- Element-wise binary ops with broadcasting ---

  private broadcastTo(other: Tensor): [Tensor, Tensor] {
    const a = this, b = other;
    if (a.shape.length === b.shape.length) {
      let same = true;
      for (let i = 0; i < a.shape.length; i++) {
        if (a.shape[i] !== b.shape[i]) { same = false; break; }
      }
      if (same) return [a, b];
    }

    let maxLen = Math.max(a.shape.length, b.shape.length);
    const aShape = [...new Array(maxLen - a.shape.length).fill(1), ...a.shape];
    const bShape = [...new Array(maxLen - b.shape.length).fill(1), ...b.shape];

    const outShape: number[] = [];
    for (let i = 0; i < maxLen; i++) {
      outShape.push(Math.max(aShape[i], bShape[i]));
    }

    const tile = (t: Tensor, sh: number[], outSh: number[]): Tensor => {
      if (sh.every((d, i) => d === outSh[i])) return t;
      const size = outSh.reduce((a, b) => a * b, 1);
      const result = new Array(size);
      const strides: number[] = [];
      strides[outSh.length - 1] = 1;
      for (let i = outSh.length - 2; i >= 0; i--) strides[i] = strides[i + 1] * outSh[i + 1];
      for (let i = 0; i < size; i++) {
        const indices: number[] = [];
        let rem = i;
        for (let d = outSh.length - 1; d >= 0; d--) {
          indices.unshift((rem % outSh[d] + outSh[d]) % outSh[d]);
          rem = Math.floor(rem / outSh[d]);
        }
        let srcIdx = 0;
        for (let d = 0; d < sh.length; d++) {
          const idx = indices[outSh.length - sh.length + d] % sh[d];
          let s = 1;
          for (let j = sh.length - 1; j >= d; j--) s *= (j === d ? 1 : outSh[outSh.length - sh.length + j]);
          srcIdx += idx * s;
        }
        result[i] = t.data[srcIdx];
      }
      const tiled = new Tensor(result, outSh, t.requires_grad);
      tiled._prev = [t];
      tiled._backward = () => {
        if (!t.requires_grad) return;
        const n = outSh.length;
        const m = sh.length;
        if (sh.every((d, i) => d === outSh[i])) {
          for (let i = 0; i < t.grad.length; i++) t.grad[i] += tiled.grad[i];
          return;
        }
        const paddedSh = [...new Array(n - m).fill(1), ...sh];
        const tStrides: number[] = [];
        tStrides[m - 1] = 1;
        for (let i = m - 2; i >= 0; i--) tStrides[i] = tStrides[i + 1] * sh[i + 1];
        for (let flat = 0; flat < tiled.grad.length; flat++) {
          let rem = flat;
          let targetFlat = 0;
          for (let d = n - 1; d >= 0; d--) {
            const idx = rem % outSh[d];
            rem = Math.floor(rem / outSh[d]);
            if (paddedSh[d] > 1) {
              targetFlat += idx * tStrides[d - (n - m)];
            }
          }
          t.grad[targetFlat] += tiled.grad[flat];
        }
      };
      return tiled;
    };

    const aTiled = tile(a, a.shape, outShape);
    const bTiled = tile(b, b.shape, outShape);
    return [aTiled, bTiled];
  }

  add(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const out = new Tensor(this.data.map(v => v + other), [...this.shape], this.requires_grad);
      out._prev = [this];
      out._backward = () => {
        if (this.requires_grad) {
          for (let i = 0; i < this.grad.length; i++) this.grad[i] += out.grad[i];
        }
      };
      return out;
    }
    const [a, b] = this.broadcastTo(other);
    const out = new Tensor(a.data.map((v, i) => v + b.data[i]), a.shape, a.requires_grad || b.requires_grad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requires_grad) {
        for (let i = 0; i < a.grad.length; i++) a.grad[i] += out.grad[i];
      }
      if (b.requires_grad) {
        for (let i = 0; i < b.grad.length; i++) b.grad[i] += out.grad[i];
      }
    };
    return out;
  }

  sub(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      return this.add(-other);
    }
    const negData = other.data.map(v => -v);
    const neg = new Tensor(negData, [...other.shape], other.requires_grad);
    neg._prev = [other];
    neg._backward = () => {
      if (other.requires_grad) {
        for (let i = 0; i < other.grad.length; i++) other.grad[i] -= neg.grad[i];
      }
    };
    return this.add(neg);
  }

  mul(other: Tensor | number): Tensor {
    if (typeof other === 'number') {
      const out = new Tensor(this.data.map(v => v * other), [...this.shape], this.requires_grad);
      out._prev = [this];
      out._backward = () => {
        if (this.requires_grad) {
          for (let i = 0; i < this.grad.length; i++) this.grad[i] += out.grad[i] * other;
        }
      };
      return out;
    }
    const [a, b] = this.broadcastTo(other);
    const out = new Tensor(a.data.map((v, i) => v * b.data[i]), a.shape, a.requires_grad || b.requires_grad);
    out._prev = [a, b];
    out._backward = () => {
      if (a.requires_grad) {
        for (let i = 0; i < a.grad.length; i++) a.grad[i] += out.grad[i] * b.data[i];
      }
      if (b.requires_grad) {
        for (let i = 0; i < b.grad.length; i++) b.grad[i] += out.grad[i] * a.data[i];
      }
    };
    return out;
  }

  div(other: number): Tensor {
    return this.mul(1 / other);
  }

  matmul(other: Tensor): Tensor {
    const a = this, b = other;
    const dims = a.shape.length, db = b.shape.length;
    if (dims < 2) throw new Error('matmul requires at least 2D');
    const M = a.shape[dims - 2], K = a.shape[dims - 1];
    const Db = b.shape[db - 1];
    if (K !== b.shape[db - 2]) throw new Error(`matmul shape mismatch ${K} vs ${b.shape[db - 2]}`);

    const aPreface = a.shape.slice(0, dims - 2);
    const bPreface = b.shape.slice(0, db - 2);
    const outPreface: number[] = [];
    const maxPrefaceLen = Math.max(aPreface.length, bPreface.length);
    for (let i = 0; i < maxPrefaceLen; i++) {
      outPreface.push(Math.max(
        aPreface[i + aPreface.length - maxPrefaceLen] ?? 1,
        bPreface[i + bPreface.length - maxPrefaceLen] ?? 1
      ));
    }

    const outShape = [...outPreface, M, Db];
    const aStrides: number[] = new Array(dims).fill(0);
    aStrides[dims - 1] = 1;
    for (let i = dims - 2; i >= 0; i--) aStrides[i] = aStrides[i + 1] * a.shape[i + 1];
    const bStrides: number[] = new Array(db).fill(0);
    bStrides[db - 1] = 1;
    for (let i = db - 2; i >= 0; i--) bStrides[i] = bStrides[i + 1] * b.shape[i + 1];
    const outStrides: number[] = new Array(outShape.length).fill(0);
    outStrides[outShape.length - 1] = 1;
    for (let i = outShape.length - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outShape[i + 1];

    const aBatchSize = aPreface.reduce((a, b) => a * b, 1) || 1;
    const bBatchSize = bPreface.reduce((a, b) => a * b, 1) || 1;
    const outBatchSize = outPreface.reduce((a, b) => a * b, 1) || 1;

    const aBatchStride = M * K;
    const bBatchStride = K * Db;
    const outBatchStride = M * Db;

    const outData = new Array(outShape.reduce((a, b) => a * b, 1)).fill(0);

    for (let batch = 0; batch < outBatchSize; batch++) {
      const ba = batch % aBatchSize;
      const bb = batch % bBatchSize;
      for (let i = 0; i < M; i++) {
        for (let j = 0; j < Db; j++) {
          let sum = 0;
          for (let k = 0; k < K; k++) {
            sum += a.data[ba * aBatchStride + i * K + k] * b.data[bb * bBatchStride + k * Db + j];
          }
          outData[batch * outBatchStride + i * Db + j] = sum;
        }
      }
    }

    const result = new Tensor(outData, outShape, a.requires_grad || b.requires_grad);
    result._prev = [this, other];

    result._backward = () => {
      if (this.requires_grad) {
        const grad = result.grad;
        for (let batch = 0; batch < outBatchSize; batch++) {
          const ba = batch % aBatchSize;
          const bb = batch % bBatchSize;
          for (let i = 0; i < M; i++) {
            for (let k = 0; k < K; k++) {
              let g = 0;
              for (let j = 0; j < Db; j++) {
                g += grad[batch * outBatchStride + i * Db + j] * other.data[bb * bBatchStride + k * Db + j];
              }
              this.grad[ba * aBatchStride + i * K + k] += g;
            }
          }
        }
      }
      if (other.requires_grad) {
        const grad = result.grad;
        for (let batch = 0; batch < outBatchSize; batch++) {
          const ba = batch % aBatchSize;
          const bb = batch % bBatchSize;
          for (let k = 0; k < K; k++) {
            for (let j = 0; j < Db; j++) {
              let g = 0;
              for (let i = 0; i < M; i++) {
                g += grad[batch * outBatchStride + i * Db + j] * this.data[ba * aBatchStride + i * K + k];
              }
              other.grad[bb * bBatchStride + k * Db + j] += g;
            }
          }
        }
      }
    };
    return result;
  }

  relu(): Tensor {
    const out = new Tensor(this.data.map(v => Math.max(0, v)), [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          if (this.data[i] > 0) this.grad[i] += out.grad[i];
        }
      }
    };
    return out;
  }

  tanh(): Tensor {
    const outData = this.data.map(v => {
      const e2 = Math.exp(2 * v);
      return (e2 - 1) / (e2 + 1);
    });
    const out = new Tensor(outData, [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          const sech2 = 1 - out.data[i] * out.data[i];
          this.grad[i] += out.grad[i] * sech2;
        }
      }
    };
    return out;
  }

  sigmoid(): Tensor {
    const sigmoid = (v: number) => 1 / (1 + Math.exp(-v));
    const out = new Tensor(this.data.map(sigmoid), [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          const s = out.data[i];
          this.grad[i] += out.grad[i] * s * (1 - s);
        }
      }
    };
    return out;
  }

  sum(axis?: number | number[]): Tensor {
    if (axis === undefined) {
      const s = this.data.reduce((a, b) => a + b, 0);
      const result = new Tensor([s], [1], this.requires_grad);
      result._prev = [this];
      result._backward = () => {
        if (this.requires_grad) {
          for (let i = 0; i < this.grad.length; i++) this.grad[i] += result.grad[0];
        }
      };
      return result;
    }
    const axes = (Array.isArray(axis) ? axis : [axis]).map(a => a < 0 ? this.shape.length + a : a);
    const n = this.shape.length;
    const sortedAxes = [...axes].sort((a, b) => b - a);
    const outShape = this.shape.filter((_, i) => !axes.includes(i));
    const outSize = outShape.length === 0 ? 1 : outShape.reduce((a, b) => a * b, 1);
    const outShapeFinal = outShape.length === 0 ? [1] : outShape;

    const outData = new Array(outSize).fill(0);
    const inStrides = this.strides();
    const outStrides: number[] = [];
    outStrides[outShapeFinal.length - 1] = 1;
    for (let i = outShapeFinal.length - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outShapeFinal[i + 1];

    for (let flat = 0; flat < this.data.length; flat++) {
      let outFlat = 0;
      let rem = flat;
      for (let d = 0; d < n; d++) {
        const idx = Math.floor(rem / inStrides[d]);
        rem = rem % inStrides[d];
        if (!axes.includes(d)) {
          const outDimIdx = outShapeFinal.length - 1 - (n - 1 - d);
          let realOutDim = outShapeFinal.length - 1;
          for (let dd = n - 1; dd >= 0; dd--) {
            if (dd === d && !axes.includes(dd)) {
              realOutDim = outShapeFinal.length - 1;
              for (let k = n - 1; k > dd; k--) if (!axes.includes(k)) realOutDim--;
              outFlat += idx * (realOutDim >= 0 ? outStrides[realOutDim] : 0);
              break;
            }
          }
        }
      }
      outData[outFlat] += this.data[flat];
    }

    const out = new Tensor(outData, outShapeFinal, this.requires_grad);
    out._prev = [this];
    const a = this;
    out._backward = () => {
      if (!a.requires_grad) return;
      for (let flat = 0; flat < a.data.length; flat++) {
        let outFlat = 0;
        let rem = flat;
        for (let d = 0; d < n; d++) {
          const idx = Math.floor(rem / inStrides[d]);
          rem = rem % inStrides[d];
          if (!axes.includes(d)) {
            let outDim = outShapeFinal.length - 1;
            for (let k = n - 1; k > d; k--) if (!axes.includes(k)) outDim--;
            outFlat += idx * outStrides[outDim];
          }
        }
        a.grad[flat] += out.grad[outFlat];
      }
    };
    return out;
  }

  mean(axis?: number | number[]): Tensor {
    const axisList = axis !== undefined ? (Array.isArray(axis) ? axis : [axis]) : undefined;
    const count = axisList
      ? axisList.reduce((acc, ax) => acc * this.shape[ax], 1)
      : this.data.length;
    return this.sum(axisList as number[]).mul(1 / count);
  }

  pow(power: number): Tensor {
    const out = new Tensor(this.data.map(v => v ** power), [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          this.grad[i] += out.grad[i] * power * this.data[i] ** (power - 1);
        }
      }
    };
    return out;
  }

  neg(): Tensor {
    return this.mul(-1);
  }

  clamp(min?: number, max?: number): Tensor {
    const out = new Tensor(this.data.map(v => {
      if (min !== undefined && v < min) return min;
      if (max !== undefined && v > max) return max;
      return v;
    }), [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          const v = this.data[i];
          if ((min === undefined || v >= min) && (max === undefined || v <= max)) {
            this.grad[i] += out.grad[i];
          }
        }
      }
    };
    return out;
  }

  abs(): Tensor {
    const out = new Tensor(this.data.map(Math.abs), [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (this.requires_grad) {
        for (let i = 0; i < this.grad.length; i++) {
          this.grad[i] += out.grad[i] * (this.data[i] >= 0 ? 1 : -1);
        }
      }
    };
    return out;
  }

  softmax(axis: number = -1): Tensor {
    const n = this.shape.length;
    axis = axis < 0 ? n + axis : axis;
    const outerSize = this.shape.slice(0, axis).reduce((a, b) => a * b, 1) || 1;
    const stride = this.shape.slice(axis + 1).reduce((a, b) => a * b, 1) || 1;
    const dim = this.shape[axis];

    const maxPerBatch = new Array(outerSize).fill(-Infinity);
    for (let o = 0; o < outerSize; o++) {
      for (let d = 0; d < dim; d++) {
        for (let s = 0; s < stride; s++) {
          const idx = o * dim * stride + d * stride + s;
          if (this.data[idx] > maxPerBatch[o]) maxPerBatch[o] = this.data[idx];
        }
      }
    }

    const exps: number[] = new Array(this.data.length);
    const sumPerBatch = new Array(outerSize).fill(0);
    for (let o = 0; o < outerSize; o++) {
      for (let d = 0; d < dim; d++) {
        for (let s = 0; s < stride; s++) {
          const idx = o * dim * stride + d * stride + s;
          const e = Math.exp(this.data[idx] - maxPerBatch[o]);
          exps[idx] = e;
          sumPerBatch[o] += e;
        }
      }
    }

    const probs = exps.map((e, i) => {
      const o = Math.floor(i / (dim * stride));
      return e / (sumPerBatch[o] + 1e-10);
    });

    const out = new Tensor(probs, [...this.shape], this.requires_grad);
    out._prev = [this];
    out._backward = () => {
      if (!this.requires_grad) return;
      for (let o = 0; o < outerSize; o++) {
        for (let d = 0; d < dim; d++) {
          for (let s = 0; s < stride; s++) {
            const idx = o * dim * stride + d * stride + s;
            let dot = 0;
            for (let d2 = 0; d2 < dim; d2++) {
              const idx2 = o * dim * stride + d2 * stride + s;
              dot += probs[idx2] * out.grad[idx2];
            }
            this.grad[idx] += probs[idx] * (out.grad[idx] - dot);
          }
        }
      }
    };
    return out;
  }

  cross_entropy(targets: number[]): Tensor {
    const batch_size = this.shape[0];
    const vocab_size = this.shape[this.shape.length - 1];
    let loss = 0;

    for (let b = 0; b < batch_size; b++) {
      const batch_offset = b * vocab_size;
      let max_logit = -Infinity;
      for (let v = 0; v < vocab_size; v++) {
        max_logit = Math.max(max_logit, this.data[batch_offset + v]);
      }
      let sum_exp = 0;
      for (let v = 0; v < vocab_size; v++) {
        sum_exp += Math.exp(this.data[batch_offset + v] - max_logit);
      }
      const target = targets[b];
      loss -= Math.log(Math.exp(this.data[batch_offset + target] - max_logit) / sum_exp + 1e-10);
    }
    loss /= batch_size;

    const result = new Tensor([loss], [1], this.requires_grad);
    result._prev = [this];
    result._backward = () => {
      if (!this.requires_grad) return;
      for (let b = 0; b < batch_size; b++) {
        const batch_offset = b * vocab_size;
        let max_logit = -Infinity;
        for (let v = 0; v < vocab_size; v++) {
          max_logit = Math.max(max_logit, this.data[batch_offset + v]);
        }
        let sum_exp = 0;
        for (let v = 0; v < vocab_size; v++) {
          sum_exp += Math.exp(this.data[batch_offset + v] - max_logit);
        }
        for (let v = 0; v < vocab_size; v++) {
          const prob = Math.exp(this.data[batch_offset + v] - max_logit) / (sum_exp + 1e-10);
          const grad_val = (prob - (v === targets[b] ? 1 : 0)) / batch_size;
          this.grad[batch_offset + v] += grad_val;
        }
      }
    };
    return result;
  }

  max(axis?: number): Tensor {
    if (axis === undefined) {
      const m = Math.max(...this.data);
      const result = new Tensor([m], [1], this.requires_grad);
      result._prev = [this];
      result._backward = () => {
        if (this.requires_grad) {
          for (let i = 0; i < this.data.length; i++) {
            if (this.data[i] === m) { this.grad[i] += result.grad[0]; break; }
          }
        }
      };
      return result;
    }
    throw new Error('max with axis not yet implemented');
  }

  // Operator aliases
  radd(other: number): Tensor { return this.add(other); }
  rmul(other: number): Tensor { return this.mul(other); }
}

export function cat(tensors: Tensor[], axis = 0): Tensor {
  const n = tensors[0].shape.length;
  axis = axis < 0 ? n + axis : axis;
  const outShape = [...tensors[0].shape];
  for (let i = 1; i < tensors.length; i++) outShape[axis] += tensors[i].shape[axis];

  const outStrides = new Array(n);
  outStrides[n - 1] = 1;
  for (let i = n - 2; i >= 0; i--) outStrides[i] = outStrides[i + 1] * outShape[i + 1];

  const outData = new Array(outShape.reduce((a, b) => a * b, 1));

  let dimOffset = 0;
  for (const t of tensors) {
    const inStrides = new Array(n);
    inStrides[n - 1] = 1;
    for (let i = n - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * t.shape[i + 1];
    for (let f = 0; f < t.data.length; f++) {
      const idx = new Array(n);
      let rem = f;
      for (let d = 0; d < n; d++) {
        idx[d] = Math.floor(rem / inStrides[d]);
        rem = rem % inStrides[d];
      }
      idx[axis] += dimOffset;
      let of = 0;
      for (let d = 0; d < n; d++) of += idx[d] * outStrides[d];
      outData[of] = t.data[f];
    }
    dimOffset += t.shape[axis];
  }

  const result = new Tensor(outData, outShape, tensors.some(t => t.requires_grad));
  result._prev = tensors;
  result._backward = () => {
    let dOff = 0;
    for (const t of tensors) {
      if (t.requires_grad) {
        const inStrides = new Array(n);
        inStrides[n - 1] = 1;
        for (let i = n - 2; i >= 0; i--) inStrides[i] = inStrides[i + 1] * t.shape[i + 1];
        for (let f = 0; f < t.data.length; f++) {
          const idx = new Array(n);
          let rem = f;
          for (let d = 0; d < n; d++) {
            idx[d] = Math.floor(rem / inStrides[d]);
            rem = rem % inStrides[d];
          }
          idx[axis] += dOff;
          let of = 0;
          for (let d = 0; d < n; d++) of += idx[d] * outStrides[d];
          t.grad[f] += result.grad[of];
        }
      }
      dOff += t.shape[axis];
    }
  };
  return result;
}