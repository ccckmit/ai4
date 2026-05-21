import { Tensor, Conv2d, MaxPool2d, Flatten, Linear, Adam, Module } from '../index';
import { DataLoader, datasets } from '../datasets';
import * as fs from 'fs';
import * as path from 'path';

class MNISTNet extends Module {
  conv1: Conv2d;
  conv2: Conv2d;
  pool1: MaxPool2d;
  pool2: MaxPool2d;
  flatten: Flatten;
  fc1: Linear;
  fc2: Linear;

  constructor() {
    super();
    this.conv1 = new Conv2d(1, 32, 3, 1, 0, true);
    this.conv2 = new Conv2d(32, 64, 3, 1, 0, true);
    this.pool1 = new MaxPool2d(2);
    this.pool2 = new MaxPool2d(2);
    this.flatten = new Flatten();
    this.fc1 = new Linear(64 * 5 * 5, 128, true);
    this.fc2 = new Linear(128, 10, true);
  }

  forward(x: Tensor): Tensor {
    let out = x.relu();
    out = this.pool1.forward(this.conv1.forward(out));
    out = out.relu();
    out = this.pool2.forward(this.conv2.forward(out));
    out = this.flatten.forward(out);
    out = out.relu();
    out = this.fc1.forward(out);
    return this.fc2.forward(out);
  }
}

export async function train() {
  console.log('Loading MNIST dataset...');
  const mnistClass = new datasets.MNIST('./data', true, true);
  const rawDataset = await mnistClass.load();

  const subsetSize = 1000;
  const subsetIndices = Array.from({ length: subsetSize }, (_, i) => i);
  const subsetData = {
    length: () => subsetSize,
    get: (indices: number[]) => rawDataset.get(indices.map((i) => subsetIndices[i])),
  };

  const trainLoader = new DataLoader(subsetData, 64, true);
  console.log(`Dataset: ${subsetSize} samples`);

  const model = new MNISTNet();
  const optimizer = new Adam(model.parameters(), 0.001);

  for (let epoch = 0; epoch < 5; epoch++) {
    let total = 0;
    let correct = 0;

    for (let batchIdx = 0; batchIdx < trainLoader.length(); batchIdx++) {
      const { xs, ys } = trainLoader.get(batchIdx);
      const batchSize = ys.length;
      const xData = xs.map((v) => (Array.isArray(v) ? v : [v])).flat();
      const images = Tensor.from([xData], true);
      images.shape = [batchSize, 1, 28, 28];
      const labels = ys;

      const logits = model.forward(images);
      const loss = logits.cross_entropy(labels);

      optimizer.zeroGrad();
      loss.backward();
      optimizer.step();

      const logitsData = logits.data;
      const vocabSize = logits.shape[logits.shape.length - 1];
      const predictions: number[] = [];
      for (let i = 0; i < batchSize; i++) {
        const offset = i * vocabSize;
        let maxVal = -Infinity;
        let maxIdx = 0;
        for (let j = 0; j < vocabSize; j++) {
          if (logitsData[offset + j] > maxVal) {
            maxVal = logitsData[offset + j];
            maxIdx = j;
          }
        }
        predictions.push(maxIdx);
      }

      total += batchSize;
      for (let i = 0; i < batchSize; i++) {
        if (predictions[i] === ys[i]) correct++;
      }

      console.log(`Epoch ${epoch + 1} Batch ${batchIdx}/${trainLoader.length()} Loss: ${loss.data[0].toFixed(4)}`);
    }

    const accuracy = (100 * correct) / total;
    console.log(`Epoch ${epoch + 1} Accuracy: ${accuracy.toFixed(2)}%`);
  }

  const dir = path.join('.', 'nn', 'mnist');
  fs.mkdirSync(dir, { recursive: true });
  const params: Record<string, number[]> = {};
  model.parameters().forEach((p, i) => {
    params[`param_${i}`] = p.data;
  });
  fs.writeFileSync(path.join(dir, 'model.npy'), JSON.stringify(params));
  console.log('Model saved to nn/mnist/model.npy');
}

train().catch(console.error);