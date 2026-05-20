import { Tensor, Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d } from '../index';

const T4 = (data: number[], shape: number[]) => new Tensor(data, shape, true);

describe('Conv2d', () => {
  test('creation', () => {
    const conv = new Conv2d(3, 16, 3);
    expect(conv.in_channels).toBe(3);
    expect(conv.out_channels).toBe(16);
    expect(conv.kernel_size).toBe(3);
  });

  test('forward shape with stride 1', () => {
    const conv = new Conv2d(3, 8, 3, 1, 0, true);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0).map(() => Math.random()), [2, 3, 10, 10]);
    const out = conv.forward(x);
    expect(out.shape).toEqual([2, 8, 8, 8]);
  });

  test('forward shape with padding', () => {
    const conv = new Conv2d(3, 8, 3, 1, 1, true);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0).map(() => Math.random()), [2, 3, 10, 10]);
    const out = conv.forward(x);
    expect(out.shape).toEqual([2, 8, 10, 10]);
  });

  test('forward shape with stride 2', () => {
    const conv = new Conv2d(3, 8, 3, 2, 0, true);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0).map(() => Math.random()), [2, 3, 10, 10]);
    const out = conv.forward(x);
    expect(out.shape[0]).toBe(2);
    expect(out.shape[1]).toBe(8);
    expect(out.shape[2]).toBe(Math.floor((10 - 3) / 2 + 1));
  });

  test('without bias', () => {
    const conv = new Conv2d(3, 8, 3, 1, 0, false);
    expect(conv.bias).toBeNull();
  });

  test('backward computes gradients', () => {
    const conv = new Conv2d(1, 1, 3, 1, 0, true);
    const x = T4(new Array(1 * 1 * 5 * 5).fill(1), [1, 1, 5, 5]);
    const out = conv.forward(x);
    const loss = out.sum();
    loss.backward();
    expect(conv.weight.grad.length).toBe(conv.weight.data.length);
  });
});

describe('MaxPool2d', () => {
  test('creation', () => {
    const pool = new MaxPool2d(2);
    expect(pool.kernel_size).toBe(2);
    expect(pool.stride).toBe(2);
  });

  test('custom stride', () => {
    const pool = new MaxPool2d(2, 1);
    expect(pool.stride).toBe(1);
  });

  test('forward shape', () => {
    const pool = new MaxPool2d(2);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0), [2, 3, 10, 10]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([2, 3, 5, 5]);
  });

  test('backward flows gradients to max positions', () => {
    const pool = new MaxPool2d(2);
    const data = [
      1, 2,
      3, 4,
    ];
    const x = new Tensor(data, [1, 1, 2, 2], true);
    const out = pool.forward(x);
    const loss = out.sum();
    loss.backward();
    expect(x.grad[3]).toBe(1);
  });
});

describe('AvgPool2d', () => {
  test('creation', () => {
    const pool = new AvgPool2d(2);
    expect(pool.kernel_size).toBe(2);
    expect(pool.stride).toBe(2);
  });

  test('forward shape', () => {
    const pool = new AvgPool2d(2);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(1), [2, 3, 10, 10]);
    const out = pool.forward(x);
    expect(out.shape).toEqual([2, 3, 5, 5]);
  });
});

describe('Flatten', () => {
  test('flattens 4D to 2D', () => {
    const flat = new Flatten();
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0), [2, 3, 10, 10]);
    const out = flat.forward(x);
    expect(out.shape).toEqual([2, 300]);
  });

  test('backward restores shape', () => {
    const flat = new Flatten();
    const x = T4(new Array(2 * 3 * 4 * 4).fill(1), [2, 3, 4, 4]);
    const out = flat.forward(x);
    const loss = out.sum();
    loss.backward();
    expect(x.grad.length).toBe(x.data.length);
  });
});

describe('BatchNorm2d', () => {
  test('creation', () => {
    const bn = new BatchNorm2d(16);
    expect(bn.num_channels).toBe(16);
    expect(bn.training).toBe(true);
  });

  test('forward shape', () => {
    const bn = new BatchNorm2d(8);
    const x = T4(new Array(4 * 8 * 10 * 10).fill(0).map(() => Math.random()), [4, 8, 10, 10]);
    const out = bn.forward(x);
    expect(out.shape).toEqual([4, 8, 10, 10]);
  });

  test('eval mode', () => {
    const bn = new BatchNorm2d(8);
    bn.eval();
    expect(bn.training).toBe(false);
  });

  test('train mode', () => {
    const bn = new BatchNorm2d(8);
    bn.eval();
    bn.train();
    expect(bn.training).toBe(true);
  });
});

describe('Dropout2d', () => {
  test('creation', () => {
    const drop = new Dropout2d(0.5);
    expect(drop.p).toBeCloseTo(0.5);
    expect(drop.training).toBe(true);
  });

  test('eval mode passes through unchanged', () => {
    const drop = new Dropout2d(0.5);
    drop.eval();
    const x = new Tensor([1, 2, 3, 4], [1, 1, 2, 2], true);
    const out = drop.forward(x);
    expect(out.data).toEqual(x.data);
  });
});

describe('CNN integration', () => {
  test('conv + relu + pool', () => {
    const conv = new Conv2d(3, 8, 3, 1, 1, true);
    const pool = new MaxPool2d(2);
    const x = T4(new Array(2 * 3 * 10 * 10).fill(0.5), [2, 3, 10, 10]);
    const out = conv.forward(x).relu();
    const pooled = pool.forward(out);
    expect(pooled.shape).toEqual([2, 8, 5, 5]);
  });

  test('simple CNN shapes', () => {
    const conv1 = new Conv2d(1, 8, 3, 1, 1, true);
    const pool1 = new MaxPool2d(2);
    const conv2 = new Conv2d(8, 16, 3, 1, 1, true);
    const pool2 = new MaxPool2d(2);
    const flat = new Flatten();

    const x = T4(new Array(4 * 1 * 28 * 28).fill(0), [4, 1, 28, 28]);

    let out = conv1.forward(x).relu();
    out = pool1.forward(out);
    expect(out.shape).toEqual([4, 8, 14, 14]);

    out = conv2.forward(out).relu();
    out = pool2.forward(out);
    expect(out.shape).toEqual([4, 16, 7, 7]);

    out = flat.forward(out);
    expect(out.shape).toEqual([4, 784]);
  });
});