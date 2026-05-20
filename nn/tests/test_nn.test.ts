import { Tensor, Module, Linear, Embedding, RMSNorm, Adam, Sequential } from '../index';

const T = (data: number[], shape: number[], rg = false) => new Tensor(data, shape, rg);

describe('Module.parameters', () => {
  test('collects parameters from Linear', () => {
    const linear = new Linear(3, 4, true);
    const params = linear.parameters();
    expect(params.length).toBe(2);
    expect(params[0]).toBe(linear.weight);
    expect(params[1]).toBe(linear.bias);
  });

  test('collects parameters from Linear without bias', () => {
    const linear = new Linear(3, 4, false);
    const params = linear.parameters();
    expect(params.length).toBe(1);
  });
});

describe('Linear', () => {
  test('forward without bias', () => {
    const linear = new Linear(3, 4, false);
    const x = T([1, 2, 3], [3], false);
    const y = linear.forward(x);
    expect(y.shape).toEqual([4]);
  });

  test('forward with bias', () => {
    const linear = new Linear(3, 4, true);
    const x = T([1, 2, 3], [3], false);
    const y = linear.forward(x);
    expect(y.shape).toEqual([4]);
    expect(y.data.length).toBe(4);
  });

  test('backward computes weight gradients', () => {
    const linear = new Linear(2, 1, true);
    const x = T([1, 2], [2], true);
    const y = linear.forward(x);
    const loss = y.sum();
    loss.backward();
    expect(linear.weight.grad).toBeDefined();
    expect(linear.weight.grad.length).toBe(2);
  });
});

describe('Embedding', () => {
  test('embeds indices', () => {
    const embed = new Embedding(10, 4);
    const indices = T([1, 3, 5], [3], false);
    const out = embed.forward(indices);
    expect(out.shape).toEqual([3, 4]);
  });

  test('has parameters', () => {
    const embed = new Embedding(10, 4);
    const params = embed.parameters();
    expect(params.length).toBe(1);
    expect(params[0]).toBe(embed.weight);
  });
});

describe('RMSNorm', () => {
  test('forward preserves shape', () => {
    const norm = new RMSNorm(4);
    const x = T([1, 2, 3, 4, 5, 6, 7, 8], [2, 4], true);
    const y = norm.forward(x);
    expect(y.shape).toEqual([2, 4]);
  });

  test('backward flows gradients', () => {
    const norm = new RMSNorm(4);
    const x = T([1, 2, 3, 4], [4], true);
    const y = norm.forward(x);
    const loss = y.sum();
    loss.backward();
    expect(x.grad).toBeDefined();
  });
});

describe('Adam', () => {
  test('step changes parameter values', () => {
    const p = T([1, 2, 3], [3], true);
    p.grad = [0.1, 0.2, 0.3];
    const optim = new Adam([p], 0.01);
    const oldData = [...p.data];
    optim.step();
    expect(p.data).not.toEqual(oldData);
  });

  test('zeroGrad resets gradients', () => {
    const p = T([1, 2, 3], [3], true);
    p.grad = [0.1, 0.2, 0.3];
    const optim = new Adam([p], 0.01);
    optim.zeroGrad();
    expect(p.grad.every(v => v === 0)).toBe(true);
  });
});

describe('Sequential', () => {
  test('chains layers', () => {
    const model = new Sequential([
      new Linear(4, 8, false),
      new Linear(8, 2, false),
    ]);
    const x = T([1, 2, 3, 4], [4], false);
    const y = model.forward(x);
    expect(y.shape).toEqual([2]);
  });
});