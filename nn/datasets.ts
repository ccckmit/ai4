import { execSync } from 'child_process';
import * as fs from 'fs';
import * as path from 'path';

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

  constructor(private root: string, private train = true, private transform?: (x: number[]) => number[]) {}

  length(): number { return this.data.length; }

  get(indices: number[]): { xs: number[]; ys: number[] } {
    const xs: number[] = [];
    const ys: number[] = [];
    for (const idx of indices) {
      const item = this.data[idx];
      xs.push(...item.xs);
      ys.push(item.ys);
    }
    return { xs: this.transform ? this.data[indices[0]].xs.map((_, i) => this.transform!(this.data.map(d => d.xs[i]).flat()).slice(0, indices.length * 784)[i] ?? 0) : this.data[indices[0]].xs, ys };
  }
}

class MNISTLoader {
  static async load(root: string, train = true, download = true): Promise<{ images: number[]; labels: number[] }> {
    const pythonScript = `
import sys
import json

try:
    import torch
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

    if ${train}:
        ds = datasets.MNIST(root="${root}", train=True, download=${download}, transform=transform)
    else:
        ds = datasets.MNIST(root="${root}", train=False, download=${download}, transform=transform)

    images = []
    labels = []
    for img, label in ds:
        images.append(img.squeeze().tolist())
        labels.append(label)

    print(json.dumps({"images": images, "labels": labels}))
except Exception as e:
    print(json.dumps({"error": str(e)}))
`;
    try {
      const out = execSync(`cd "${root}" && python3 -c "${pythonScript.replace(/"/g, '\\"').replace(/\n/g, ' ')}"`, { timeout: 60000 });
      return JSON.parse(out.toString());
    } catch (e) {
      return { images: [], labels: [] };
    }
  }
}

export const datasets = {
  MNIST: class {
    constructor(private root: string, train = true, download = true) {}
  }
};