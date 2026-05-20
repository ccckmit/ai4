import { Tensor, Conv2d, MaxPool2d, Flatten, Linear, Adam, Module } from '../index';

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
    out = this.pool1(this.conv1(out));
    out = out.relu();
    out = this.pool2(this.conv2(out));
    out = this.flatten(out);
    out = out.relu();
    out = this.fc1(out);
    return this.fc2(out);
  }
}

export async function train() {
  console.log('MNIST training requires the nn/datasets.ts MNIST loader.');
  console.log('Use the Python nn/mnist/train.py instead:');
  console.log('  PYTHONPATH=. uv run python nn/mnist/train.py');
}