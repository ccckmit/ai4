import { execSync } from 'child_process';
import * as fs from 'fs';

export class DataLoader<T = number[][]> {
  dataset: Dataset<T>;
  batch_size: number;
  shuffle: boolean;
  indices: number[];

  constructor(dataset: Dataset<T>, batch_size = 32, shuffle = true) {
    this.dataset = dataset;
    this.batch_size = batch_size;
    this.shuffle = shuffle;
    this.indices = Array.from({ length: dataset.length() }, (_, i) => i);
    if (shuffle) this._shuffle();
  }

  private _shuffle() {
    for (let i = this.indices.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [this.indices[i], this.indices[j]] = [this.indices[j], this.indices[i]];
    }
  }

  length(): number {
    return Math.ceil(this.dataset.length() / this.batch_size);
  }

  get(batchIdx: number): { xs: T; ys: number[] } {
    const start = batchIdx * this.batch_size;
    const end = Math.min(start + this.batch_size, this.dataset.length());
    const batchIdxArr = this.indices.slice(start, end);
    return this.dataset.get(batchIdxArr);
  }
}

export interface Dataset<T = number[][]> {
  length(): number;
  get(indices: number[]): { xs: T; ys: number[] };
}

export class MnistDataset implements Dataset<number[]> {
  private data: { xs: number[]; ys: number }[] = [];

  constructor(
    private root: string,
    private train = true,
    private transform?: (x: number[]) => number[]
  ) {}

  async load(): Promise<void> {
    const scriptPath = '/tmp/mnist_load.py';
    const outPath = '/tmp/mnist_out.json';
    const pythonScript = `import sys
import json
import os
os.makedirs('${this.root}', exist_ok=True)

try:
    import torch
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if ${this.train ? 'True' : 'False'}:
        ds = datasets.MNIST(root="${this.root}", train=True, download=True, transform=transform)
    else:
        ds = datasets.MNIST(root="${this.root}", train=False, download=True, transform=transform)

    images = []
    labels = []
    for img, label in ds:
        images.append(img.squeeze().tolist())
        labels.append(label)

    with open('${outPath}', 'w') as f:
        json.dump({"images": images, "labels": labels}, f)
    print("DONE")
except Exception as e:
    with open('${outPath}', 'w') as f:
        json.dump({"error": str(e)}, f)
    print("ERROR")
`;
    fs.writeFileSync(scriptPath, pythonScript);
    try {
      execSync(`python3 "${scriptPath}"`, { timeout: 300000 });
      const content = fs.readFileSync(outPath, 'utf-8');
      const result = JSON.parse(content);
      if (result.error) {
        console.error('MNIST load error:', result.error);
        return;
      }
      for (let i = 0; i < result.labels.length; i++) {
        const xs = result.images[i];
        const ys = result.labels[i];
        let transformedXs = xs;
        if (this.transform) {
          transformedXs = this.transform(xs);
        }
        this.data.push({ xs: transformedXs, ys });
      }
    } catch (e) {
      console.error('Failed to load MNIST:', e);
    } finally {
      try { fs.unlinkSync(scriptPath); } catch {}
      try { fs.unlinkSync(outPath); } catch {}
    }
  }

  length(): number {
    return this.data.length;
  }

  get(indices: number[]): { xs: number[]; ys: number[] } {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const idx of indices) {
      const item = this.data[idx];
      xs.push(...item.xs);
      ys.push(item.ys);
    }
    return { xs, ys };
  }
}

export class Compose {
  transforms: ((x: number[]) => number[])[];

  constructor(transforms: ((x: number[]) => number[])[]) {
    this.transforms = transforms;
  }

  __call__(x: number[]): number[] {
    let result = x;
    for (const t of this.transforms) {
      result = t(result);
    }
    return result;
  }
}

export function ToTensor(): (x: number[] | number[][]) => number[] {
  return (x) => (Array.isArray(x[0]) ? (x as number[][]).flat() : x as number[]);
}

export function Normalize(mean: number[], std: number[]): (x: number[]) => number[] {
  return (x: number[]) => x.map((v, i) => (v - mean[i % mean.length]) / std[i % std.length]);
}

export const datasets = {
  MNIST: class {
    private dataset: MnistDataset;

    constructor(root: string, train = true, download = true) {
      const transforms: ((x: number[]) => number[])[] = [
        ToTensor(),
        Normalize([0.5], [0.5]),
      ];
      const composed = (x: number[]) => {
        let result = x;
        for (const t of transforms) {
          result = t(result);
        }
        return result;
      };
      this.dataset = new MnistDataset(root, train, composed);
    }

    async load(): Promise<MnistDataset> {
      await this.dataset.load();
      return this.dataset;
    }
  }
};