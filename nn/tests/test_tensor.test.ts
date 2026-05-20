import { Tensor, cat } from '../tensor';

const T = (data: number[], shape: number[], rg = false) => new Tensor(data, shape, rg);

describe('Tensor creation', () => {
  test('creates tensor with correct shape', () => {
    const t = T([1, 2, 3, 4], [2, 2], true);
    expect(t.shape).toEqual([2, 2]);
    expect(t.data).toEqual([1, 2, 3, 4]);
    expect(t.requires_grad).toBe(true);
  });

  test('zeros factory', () => {
    const t = Tensor.zeros([3, 4]);
    expect(t.shape).toEqual([3, 4]);
    expect(t.data.every(v => v === 0)).toBe(true);
  });

  test('ones factory', () => {
    const t = Tensor.ones([2, 3]);
    expect(t.shape).toEqual([2, 3]);
    expect(t.data.every(v => v === 1)).toBe(true);
  });

  test('randn factory', () => {
    const t = Tensor.randn([10, 5]);
    expect(t.shape).toEqual([10, 5]);
  });
});

describe('Tensor add', () => {
  test('adds two tensors element-wise', () => {
    const a = T([1, 2, 3, 4], [2, 2], true);
    const b = T([1, 1, 1, 1], [2, 2], true);
    const c = a.add(b);
    expect(c.data).toEqual([2, 3, 4, 5]);
  });

  test('add scalar broadcasts', () => {
    const a = T([1, 2, 3, 4], [2, 2], true);
    const c = a.add(1);
    expect(c.data).toEqual([2, 3, 4, 5]);
  });
});

describe('Tensor mul', () => {
  test('multiplies element-wise', () => {
    const a = T([2, 3, 4, 5], [2, 2], true);
    const b = T([1, 2, 3, 4], [2, 2], true);
    const c = a.mul(b);
    expect(c.data).toEqual([2, 6, 12, 20]);
  });

  test('mul scalar broadcasts', () => {
    const a = T([1, 2, 3, 4], [2, 2], true);
    const c = a.mul(2);
    expect(c.data).toEqual([2, 4, 6, 8]);
  });
});

describe('Tensor matmul', () => {
  test('2x2 matrix multiply', () => {
    const a = T([1, 2, 3, 4], [2, 2], true);
    const b = T([5, 6, 7, 8], [2, 2], true);
    const c = a.matmul(b);
    expect(c.data[0]).toBe(19);
    expect(c.data[1]).toBe(22);
    expect(c.data[2]).toBe(43);
    expect(c.data[3]).toBe(50);
  });

  test('3x2 @ 2x4', () => {
    const a = T([1, 2, 3, 4, 5, 6], [3, 2]);
    const b = T([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    const c = a.matmul(b);
    expect(c.shape).toEqual([3, 4]);
  });
});

describe('Tensor backward', () => {
  test('computes gradients for mul', () => {
    const a = T([1, 2, 3], [3], true);
    a.grad = [0, 0, 0];
    const b = T([1, 1, 1], [3], true);
    b.grad = [0, 0, 0];
    const c = a.mul(b);
    const loss = c.sum();
    loss.backward();
    expect(a.grad).toEqual([1, 1, 1]);
    expect(b.grad).toEqual([1, 2, 3]);
  });

  test('computes gradients for matmul', () => {
    // a=[1,1,1,1] (2x2), b=[1,1,1,1] (2x2), c=a@b=[[2,2],[2,2]]
    // dL/da = dL/dc @ b^T = ones(2,2) @ ones(2,2) = [[2,2],[2,2]]
    const a = T([1, 1, 1, 1], [2, 2], true);
    a.grad = [0, 0, 0, 0];
    const b = T([1, 1, 1, 1], [2, 2], true);
    b.grad = [0, 0, 0, 0];
    const c = a.matmul(b);
    const loss = c.sum();
    loss.backward();
    expect(a.grad).toEqual([2, 2, 2, 2]);
    expect(b.grad).toEqual([2, 2, 2, 2]);
  });
});

describe('Tensor relu', () => {
  test('zeroes negative values', () => {
    const x = T([-1, 0, 1, 2], [4], true);
    const y = x.relu();
    expect(y.data).toEqual([0, 0, 1, 2]);
  });

  test('backward flows only through positive inputs', () => {
    const x = T([-1, 0, 1, 2], [4], true);
    const y = x.relu();
    const loss = y.sum();
    x.zeroGrad();
    loss.backward();
    expect(x.grad).toEqual([0, 0, 1, 1]);
  });
});

describe('Tensor softmax', () => {
  test('sums to 1', () => {
    const x = T([1, 2, 3], [3], true);
    const y = x.softmax(0);
    const sum = y.data.reduce((a, b) => a + b, 0);
    expect(sum).toBeCloseTo(1, 2);
  });
});

describe('Tensor transpose', () => {
  test('2x3 becomes 3x2', () => {
    const x = T([1, 2, 3, 4, 5, 6], [2, 3], true);
    const y = x.transpose(0, 1);
    expect(y.shape).toEqual([3, 2]);
  });
});

describe('cat', () => {
  test('concatenates along axis 0', () => {
    const a = T([1, 2, 3], [3], true);
    const b = T([4, 5], [2], true);
    const c = cat([a, b], 0);
    expect(c.data).toEqual([1, 2, 3, 4, 5]);
  });
});

describe('Tensor reshape', () => {
  test('flattens 2x3 to 1x6', () => {
    const x = T([1, 2, 3, 4, 5, 6], [2, 3], true);
    const y = x.reshape(6);
    expect(y.shape).toEqual([6]);
    expect(y.data).toEqual([1, 2, 3, 4, 5, 6]);
  });
});

describe('Tensor sum', () => {
  test('sums all elements', () => {
    const x = T([1, 2, 3, 4], [4], true);
    const s = x.sum();
    expect(s.data[0]).toBe(10);
  });
});

describe('Tensor neg', () => {
  test('negates values', () => {
    const x = T([1, -2, 3], [3], true);
    const y = x.neg();
    expect(y.data).toEqual([-1, 2, -3]);
  });
});

describe('Tensor pow', () => {
  test('squares values', () => {
    const x = T([2, 3], [2], true);
    const y = x.pow(2);
    expect(y.data).toEqual([4, 9]);
  });
});

describe('Tensor mean', () => {
  test('computes mean', () => {
    const x = T([1, 2, 3, 4], [4], true);
    const m = x.mean();
    expect(m.data[0]).toBe(2.5);
  });
});