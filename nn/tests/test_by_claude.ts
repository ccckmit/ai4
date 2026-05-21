/**
 * Comprehensive test suite for nn module.
 * Covers forward values, backward (gradient) correctness via numerical
 * finite-difference comparison, and edge cases — mirrors test_by_claude.py
 */
import { Tensor, cat } from '../tensor';
import {
  Module, Linear, Embedding, RMSNorm, Adam,
  Sequential, ReLU, Tanh, mse_loss,
} from '../nn';
import { GPT } from '../gpt';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const ATOL = 1e-5;
const RTOL = 5e-2;
const EPS = 1e-3;

function assert(cond: boolean, msg: string) {
  if (!cond) throw new Error(`Assertion failed: ${msg}`);
}

function assertAlmost(a: number, b: number, tol = ATOL) {
  if (Math.abs(a - b) > tol) throw new Error(`Expected ${a} ≈ ${b}`);
}

function assertAllClose(actual: number[], expected: number[], tol = ATOL) {
  if (actual.length !== expected.length) throw new Error(`Length mismatch: ${actual.length} vs ${expected.length}`);
  for (let i = 0; i < actual.length; i++) {
    if (Math.abs(actual[i] - expected[i]) > tol) {
      throw new Error(`Mismatch at [${i}]: ${actual[i]} ≈ ${expected[i]}`);
    }
  }
}

function numericalGrad(f: () => Tensor, p: Tensor, eps = EPS): number[] {
  const grad: number[] = new Array(p.data.length).fill(0);
  for (let i = 0; i < p.data.length; i++) {
    const orig = p.data[i];
    p.data[i] = orig + eps;
    const fp = f().data.reduce((a, b) => a + b, 0);
    p.data[i] = orig - eps;
    const fm = f().data.reduce((a, b) => a + b, 0);
    p.data[i] = orig;
    grad[i] = (fp - fm) / (2 * eps);
  }
  return grad;
}

function gradCheck(
  makeFn: () => [{ name: string; tensor: Tensor }[], () => Tensor],
  tol = RTOL,
) {
  const [params, f] = makeFn();
  for (const { name, tensor: p } of params) {
    // Analytical
    for (const { tensor: pp } of params) pp.zeroGrad();
    const out = f();
    out.backward();
    const analytic = [...p.grad];

    // Numerical
    for (const { tensor: pp } of params) pp.zeroGrad();
    const numeric = numericalGrad(f, p);

    let maxRelErr = 0;
    for (let i = 0; i < analytic.length; i++) {
      const denom = Math.abs(numeric[i]) + Math.abs(analytic[i]) + 1e-6;
      const relErr = Math.abs(numeric[i] - analytic[i]) / denom;
      if (relErr > maxRelErr) maxRelErr = relErr;
    }

    if (maxRelErr >= tol) {
      throw new Error(
        `Gradient check FAILED for '${name}': max_rel_err=${maxRelErr.toExponential(4)} (tol=${tol})\n` +
        `  numeric  = [${numeric.slice(0, 6).map(v => v.toFixed(4)).join(', ')}]\n` +
        `  analytic = [${analytic.slice(0, 6).map(v => v.toFixed(4)).join(', ')}]`,
      );
    }

    console.log(`  [PASS] grad_check '${name}' (max_rel_err=${maxRelErr.toExponential(4)})`);
  }
}

function randn(n: number): number[] {
  const a = new Array(n);
  for (let i = 0; i < n; i++) {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    a[i] = Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  return a;
}

function randnShape(shape: number[], requires_grad = false): Tensor {
  const size = shape.reduce((a, b) => a * b, 1);
  return new Tensor(randn(size), shape, requires_grad);
}

// ---------------------------------------------------------------------------
// 1. Arithmetic ops
// ---------------------------------------------------------------------------

function testAddForward() {
  const a = Tensor.from([1, 2, 3]);
  const b = Tensor.from([4, 5, 6]);
  assertAllClose((a.add(b)).data, [5, 7, 9]);
}

function testAddGrad() {
  gradCheck(() => {
    const a = new Tensor([1, 2, 3], [3], true);
    const b = new Tensor([4, 5, 6], [3], true);
    return [[{ name: 'a', tensor: a }, { name: 'b', tensor: b }], () => a.add(b)];
  });
}

function testAddScalar() {
  const x = new Tensor([1, 2], [2], true);
  const out = x.add(3);
  assertAllClose(out.data, [4, 5]);
}

function testMulForward() {
  const a = Tensor.from([2, 3]);
  const b = Tensor.from([4, -1]);
  assertAllClose((a.mul(b)).data, [8, -3]);
}

function testMulGrad() {
  gradCheck(() => {
    const a = new Tensor([2, -1, 0.5], [3], true);
    const b = new Tensor([3, 4, -2], [3], true);
    return [[{ name: 'a', tensor: a }, { name: 'b', tensor: b }], () => a.mul(b)];
  });
}

function testSubForward() {
  const a = Tensor.from([5, 3]);
  const b = Tensor.from([2, 4]);
  assertAllClose(a.sub(b).data, [3, -1]);
}

function testNeg() {
  const x = new Tensor([1, -2], [2], true);
  const out = x.neg();
  assertAllClose(out.data, [-1, 2]);
}

function testPowForward() {
  const x = Tensor.from([2, 3]);
  assertAllClose(x.pow(2).data, [4, 9]);
}

function testPowGrad() {
  gradCheck(() => {
    const x = new Tensor([1, 2, -1], [3], true);
    return [[{ name: 'x', tensor: x }], () => x.pow(3)];
  });
}

function testMatmulForward() {
  const a = Tensor.from([[1, 2], [3, 4]]);
  const b = Tensor.from([[1, 0], [0, 1]]);
  assertAllClose(a.matmul(b).data, [1, 2, 3, 4]);
}

function testMatmulGrad() {
  gradCheck(() => {
    const a = randnShape([3, 4], true);
    const b = randnShape([4, 2], true);
    return [[{ name: 'A', tensor: a }, { name: 'B', tensor: b }], () => a.matmul(b)];
  });
}

function testBatchedMatmulGrad() {
  gradCheck(() => {
    const a = randnShape([2, 3, 4], true);
    const b = randnShape([2, 4, 5], true);
    return [[{ name: 'A', tensor: a }, { name: 'B', tensor: b }], () => a.matmul(b)];
  });
}

function testBroadcastAddGrad() {
  gradCheck(() => {
    const a = randnShape([3, 4], true);
    const b = randnShape([4], true);
    return [[{ name: 'a', tensor: a }, { name: 'b', tensor: b }], () => a.add(b)];
  });
}

// ---------------------------------------------------------------------------
// 2. Reduction ops
// ---------------------------------------------------------------------------

function testSumGlobalForward() {
  const x = Tensor.from([[1, 2], [3, 4]]);
  assertAlmost(x.sum().data[0], 10);
}

function testSumGlobalGrad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.sum()];
  });
}

function testSumAxis0Forward() {
  const x = Tensor.from([[1, 2], [3, 4]]);
  assert(x.sum(0).shape[0] === 2, 'sum(0) should keep dim 0');
  assertAllClose(x.sum(0).data, [4, 6]);
}

function testSumAxis0Grad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.sum(0)];
  });
}

function testSumAxis1Forward() {
  const x = Tensor.from([[1, 2, 3], [4, 5, 6]]);
  assertAllClose(x.sum(1).data, [6, 15]);
}

function testSumAxis1Grad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.sum(1)];
  });
}

function testMeanGlobalGrad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.mean()];
  });
}

function testMeanAxis1Grad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.mean(1)];
  });
}

function testMeanAxisValue() {
  const x = Tensor.from([[2, 4], [6, 8]]);
  assertAllClose(x.mean(1).data, [3, 7]);
}

// ---------------------------------------------------------------------------
// 3. Activation functions
// ---------------------------------------------------------------------------

function testReluForward() {
  const x = Tensor.from([-1, 0, 2]);
  assertAllClose(x.relu().data, [0, 0, 2]);
}

function testReluGrad() {
  gradCheck(() => {
    const x = new Tensor([-1, 0.5, 2, -0.1], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.relu()];
  });
}

function testTanhForward() {
  const x = Tensor.from([0]);
  assertAlmost(x.tanh().data[0], 0, ATOL);
}

function testTanhGrad() {
  gradCheck(() => {
    const x = new Tensor([-1, 0, 1, 2], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.tanh()];
  });
}

function testSoftmaxSumsToOne() {
  const x = randnShape([4, 6]);
  const s = x.softmax();
  for (let i = 0; i < 4; i++) {
    let rowSum = 0;
    for (let j = 0; j < 6; j++) rowSum += s.data[i * 6 + j];
    assertAlmost(rowSum, 1, 1e-5);
  }
}

function testSoftmaxGrad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    const w = randnShape([3, 4]);
    return [[{ name: 'x', tensor: x }], () => {
      const s = x.softmax();
      const m = s.mul(w);
      return m.sum();
    }];
  });
}

// ---------------------------------------------------------------------------
// 4. Shape ops
// ---------------------------------------------------------------------------

function testReshapeForward() {
  const x = new Tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], [3, 4]);
  assert(x.reshape(2, 6).shape[0] === 2 && x.reshape(2, 6).shape[1] === 6, 'reshape shape');
}

function testReshapeGrad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.reshape(2, 6)];
  });
}

function testTransposeForward() {
  const x = Tensor.from([[1, 2], [3, 4]]);
  const out = x.transpose(0, 1);
  assertAllClose(out.data, [1, 3, 2, 4]);
}

function testTransposeGrad() {
  gradCheck(() => {
    const x = randnShape([3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.transpose(0, 1)];
  });
}

function testTranspose4dGrad() {
  gradCheck(() => {
    const x = randnShape([1, 4, 3, 4], true);
    return [[{ name: 'x', tensor: x }], () => x.transpose(1, 2)];
  });
}

function testCatForward() {
  const a = Tensor.from([[1, 2], [3, 4]]);
  const b = Tensor.from([[5, 6, 7], [8, 9, 10]]);
  const out = cat([a, b], 1);
  assert(out.shape[0] === 2 && out.shape[1] === 5, 'cat shape');
  assertAllClose(out.data.slice(0, 5), [1, 2, 5, 6, 7]);
  assertAllClose(out.data.slice(5), [3, 4, 8, 9, 10]);
}

function testCatAxis1Grad() {
  gradCheck(() => {
    const a = randnShape([2, 3], true);
    const b = randnShape([2, 4], true);
    return [[{ name: 'a', tensor: a }, { name: 'b', tensor: b }], () => cat([a, b], 1)];
  });
}

function testCatAxis0Grad() {
  gradCheck(() => {
    const a = randnShape([3, 2], true);
    const b = randnShape([2, 2], true);
    return [[{ name: 'a', tensor: a }, { name: 'b', tensor: b }], () => cat([a, b], 0)];
  });
}

// ---------------------------------------------------------------------------
// 5. Clamping
// ---------------------------------------------------------------------------

function testClampBothForward() {
  const x = Tensor.from([-3, 0.5, 3]);
  assertAllClose(x.clamp(-1, 1).data, [-1, 0.5, 1]);
}

function testClampBothGrad() {
  gradCheck(() => {
    const x = new Tensor([-2, -0.5, 0.5, 2], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.clamp(-1, 1)];
  });
}

function testClampMinOnlyGrad() {
  gradCheck(() => {
    const x = new Tensor([-2, -0.5, 0.5, 2], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.clamp(0)];
  });
}

function testClampMinOnlyGradValue() {
  const x = new Tensor([-2, 1, 3], [3], true);
  x.clamp(0).sum().backward();
  assertAlmost(x.grad[0], 0, ATOL);
  assertAlmost(x.grad[1], 1, ATOL);
  assertAlmost(x.grad[2], 1, ATOL);
}

function testClampMaxOnlyGrad() {
  gradCheck(() => {
    const x = new Tensor([-2, -0.5, 0.5, 2], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.clamp(undefined, 0)];
  });
}

function testAbsForward() {
  const x = Tensor.from([-3, 0, 2]);
  assertAllClose(x.abs().data, [3, 0, 2]);
}

function testAbsGrad() {
  gradCheck(() => {
    const x = new Tensor([-2, -0.5, 0.5, 2], [4], true);
    return [[{ name: 'x', tensor: x }], () => x.abs()];
  });
}

// ---------------------------------------------------------------------------
// 6. Loss functions
// ---------------------------------------------------------------------------

function testMseLossForward() {
  const pred = Tensor.from([1, 2, 3]);
  const target = Tensor.from([1, 2, 3]);
  assertAlmost(mse_loss(pred, target).data[0], 0, ATOL);
}

function testMseLossForwardValue() {
  const pred = Tensor.from([0, 0]);
  const target = Tensor.from([1, 3]);
  assertAlmost(mse_loss(pred, target).data[0], 5, 1e-4);
}

function testMseLossGrad() {
  gradCheck(() => {
    const pred = new Tensor([1, 2, 3], [3], true);
    const target = new Tensor([1.5, 2.5, 3.5], [3], false);
    return [[{ name: 'pred', tensor: pred }], () => mse_loss(pred, target)];
  });
}

function testMseLossGradNotZero() {
  const pred = new Tensor([1, 2, 3], [3], true);
  const target = new Tensor([1.5, 2.5, 3.5], [3], false);
  mse_loss(pred, target).backward();
  assert(pred.grad.some(v => v !== 0), 'grad must not be all-zero');
}

function testMseLossGradValue() {
  const pred = new Tensor([1, 2, 3], [3], true);
  const target = new Tensor([1.5, 2.5, 3.5], [3], false);
  mse_loss(pred, target).backward();
  const expected = pred.data.map((v, i) => 2 * (v - target.data[i]) / 3);
  assertAllClose(pred.grad, expected);
}

function testCrossEntropyForwardNonneg() {
  const logits = randnShape([2, 5], true);
  const targets = [1, 2, 0, 3, 4, 1];
  const loss = logits.cross_entropy(targets);
  assert(loss.data[0] > 0, 'CE loss must be positive');
}

function testCrossEntropyGrad() {
  const targets = [1, 2, 0, 3, 4, 1];
  gradCheck(() => {
    const logits = randnShape([2, 5], true);
    return [[{ name: 'logits', tensor: logits }], () => logits.cross_entropy(targets)];
  });
}

// ---------------------------------------------------------------------------
// 7. Layers
// ---------------------------------------------------------------------------

function testLinearForwardShape() {
  const layer = new Linear(4, 8);
  const x = randnShape([3, 4]);
  assert(layer.forward(x).shape[1] === 8, 'linear output dim');
}

function testLinearBiasForward() {
  const layer = new Linear(2, 3, true);
  if (layer.bias) {
    layer.bias.data.fill(1);
    layer.bias.grad = new Array(layer.bias.data.length).fill(0);
  }
  layer.weight.data.fill(0);
  layer.weight.grad = new Array(layer.weight.data.length).fill(0);
  const x = Tensor.from([[1, 1]]);
  assertAllClose(layer.forward(x).data, [1, 1, 1]);
}

function testLinearGrad() {
  gradCheck(() => {
    const layer = new Linear(4, 3, true);
    const x = randnShape([2, 4], true);
    const params = [{ name: 'x', tensor: x }, { name: 'W', tensor: layer.weight }];
    if (layer.bias) params.push({ name: 'b', tensor: layer.bias });
    return [params, () => layer.forward(x)];
  });
}

function testEmbeddingForwardShape() {
  const emb = new Embedding(10, 4);
  const indices = Tensor.from([[1, 3, 5]]);
  const out = emb.forward(indices);
  assert(out.shape[0] === 1 && out.shape[1] === 3 && out.shape[2] === 4, 'embedding shape');
}

function testEmbeddingForwardValues() {
  const emb = new Embedding(10, 4);
  const idx = Tensor.from([[2, 5]]);
  const out = emb.forward(idx);
  for (let j = 0; j < 4; j++) assertAlmost(out.data[0 * 4 + j], emb.weight.data[2 * 4 + j]);
  for (let j = 0; j < 4; j++) assertAlmost(out.data[1 * 4 + j], emb.weight.data[5 * 4 + j]);
}

function testEmbeddingGrad() {
  gradCheck(() => {
    const emb = new Embedding(10, 4);
    const idx = Tensor.from([[1, 3, 1, 5]]);
    return [[{ name: 'weight', tensor: emb.weight }], () => emb.forward(idx)];
  });
}

function testEmbeddingDuplicateIndicesGrad() {
  const emb = new Embedding(5, 3);
  const idx = Tensor.from([[1, 1]]);
  emb.weight.zeroGrad();
  emb.forward(idx).sum().backward();
  assertAllClose(
    emb.weight.grad.slice(1 * 3, 1 * 3 + 3),
    [2, 2, 2],
  );
}

function testRmsnormOutputShape() {
  const norm = new RMSNorm(8);
  const x = randnShape([4, 8], true);
  assert(norm.forward(x).shape[1] === 8, 'rmsnorm output dim');
}

function testRmsnormRmsEqualsOne() {
  const norm = new RMSNorm(16, 0);
  const x = randnShape([4, 16]);
  const out = norm.forward(x);
  for (let i = 0; i < 4; i++) {
    let sumSq = 0;
    for (let j = 0; j < 16; j++) sumSq += out.data[i * 16 + j] ** 2;
    const rms = Math.sqrt(sumSq / 16);
    assertAlmost(rms, 1, 1e-4);
  }
}

function testRmsnormGrad() {
  gradCheck(() => {
    const x = randnShape([2, 4], true);
    const norm = new RMSNorm(4);
    return [[{ name: 'x', tensor: x }], () => norm.forward(x)];
  });
}

function testRmsnorm3dGrad() {
  gradCheck(() => {
    const x = randnShape([1, 3, 16], true);
    const norm = new RMSNorm(16);
    return [[{ name: 'x', tensor: x }], () => norm.forward(x)];
  });
}

function testSequentialForward() {
  const model = new Sequential([new Linear(4, 8), new ReLU(), new Linear(8, 2)]);
  const x = randnShape([3, 4]);
  assert(model.forward(x).shape[1] === 2, 'sequential output dim');
}

function testReluModule() {
  const x = Tensor.from([-1, 2]);
  assertAllClose(new ReLU().forward(x).data, [0, 2]);
}

function testTanhModule() {
  const x = Tensor.from([0]);
  assertAlmost(new Tanh().forward(x).data[0], 0, ATOL);
}

// ---------------------------------------------------------------------------
// 8. Module.parameters()
// ---------------------------------------------------------------------------

function testLinearParameters() {
  const layer = new Linear(4, 8);
  assert(layer.parameters().includes(layer.weight), 'weight in params');
}

function testLinearBiasParameters() {
  const layer = new Linear(4, 8, true);
  const params = layer.parameters();
  assert(params.includes(layer.weight), 'weight');
  if (layer.bias) assert(params.includes(layer.bias), 'bias');
}

function testSequentialParameters() {
  const model = new Sequential([new Linear(4, 8, true), new Linear(8, 2, true)]);
  assert(model.parameters().length === 4, '4 params for 2 Linear with bias');
}

function testRmsnormScaleNotInParams() {
  const norm = new RMSNorm(4);
  assert(!norm.parameters().includes(norm.scale), 'scale should not be in params');
}

// ---------------------------------------------------------------------------
// 9. Adam optimizer
// ---------------------------------------------------------------------------

function testAdamStepDecreasesLoss() {
  const pred = new Tensor([0, 0, 0], [3], true);
  const target = new Tensor([1, 2, 3], [3], false);
  const opt = new Adam([pred], 0.1);
  const lossBefore = mse_loss(pred, target).data[0];
  for (let i = 0; i < 20; i++) {
    opt.zeroGrad();
    mse_loss(pred, target).backward();
    opt.step();
  }
  const lossAfter = mse_loss(pred, target).data[0];
  assert(lossAfter < lossBefore, 'Adam did not reduce loss');
}

function testAdamZeroGrad() {
  const p = new Tensor([1, 2], [2], true);
  const opt = new Adam([p], 0.01);
  p.mul(Tensor.from([1, 1])).sum().backward();
  assert(p.grad.some((v: number) => v !== 0), 'grad should be non-zero');
  opt.zeroGrad();
  assertAllClose(p.grad, [0, 0]);
}

function testAdamConvergesLinearRegression() {
  const layer = new Linear(1, 1, true);
  const opt = new Adam(layer.parameters(), 0.05);
  const flatX: number[] = [];
  const flatY: number[] = [];
  for (let i = 0; i < 20; i++) {
    const xv = -1 + i * (2 / 19);
    flatX.push(xv);
    flatY.push(2 * xv);
  }
  const xT = new Tensor(flatX, [20, 1], false);
  const yT = new Tensor(flatY, [20, 1], false);
  for (let iter = 0; iter < 200; iter++) {
    opt.zeroGrad();
    const pred = layer.forward(xT);
    const loss = mse_loss(pred, yT);
    loss.backward();
    opt.step();
    if (loss.data[0] < 0.01) break;
  }
  const finalLoss = mse_loss(layer.forward(xT), yT).data[0];
  assert(finalLoss < 0.01, `Loss too high: ${finalLoss.toFixed(4)}`);
}

// ---------------------------------------------------------------------------
// 10. Backward graph integrity
// ---------------------------------------------------------------------------

function testZeroGradResets() {
  const x = new Tensor([1, 2], [2], true);
  x.mul(x).sum().backward();
  assert(x.grad.some((v: number) => v !== 0), 'grad non-zero');
  x.zeroGrad();
  assertAllClose(x.grad, [0, 0]);
}

function testGradAccumulates() {
  const x = new Tensor([3], [1], true);
  x.mul(x).sum().backward();
  const g1 = [...x.grad];
  x.mul(x).sum().backward();
  assertAllClose(x.grad, g1.map((v: number) => 2 * v));
}

function testChainRule() {
  gradCheck(() => {
    const x = new Tensor([-0.5, 0.5, 1], [3], true);
    const one = new Tensor([1], [1], false);
    return [[{ name: 'x', tensor: x }], () => x.pow(2).add(one).tanh()];
  });
}

function testDeepChain() {
  gradCheck(() => {
    const x = randnShape([2, 4], true);
    const W = randnShape([4, 3], true);
    const b = randnShape([3], true);
    return [[{ name: 'x', tensor: x }, { name: 'W', tensor: W }, { name: 'b', tensor: b }],
      () => x.matmul(W).add(b).relu()];
  });
}

// ---------------------------------------------------------------------------
// Run all tests
// ---------------------------------------------------------------------------

const tests = [
  testAddForward,
  testAddGrad,
  testAddScalar,
  testMulForward,
  testMulGrad,
  testSubForward,
  testNeg,
  testPowForward,
  testPowGrad,
  testMatmulForward,
  testMatmulGrad,
  testBatchedMatmulGrad,
  testBroadcastAddGrad,
  testSumGlobalForward,
  testSumGlobalGrad,
  testSumAxis0Forward,
  testSumAxis0Grad,
  testSumAxis1Forward,
  testSumAxis1Grad,
  testMeanGlobalGrad,
  testMeanAxis1Grad,
  testMeanAxisValue,
  testReluForward,
  testReluGrad,
  testTanhForward,
  testTanhGrad,
  testSoftmaxSumsToOne,
  testSoftmaxGrad,
  testReshapeForward,
  testReshapeGrad,
  testTransposeForward,
  testTransposeGrad,
  testTranspose4dGrad,
  testCatForward,
  testCatAxis1Grad,
  testCatAxis0Grad,
  testClampBothForward,
  testClampBothGrad,
  testClampMinOnlyGrad,
  testClampMinOnlyGradValue,
  testClampMaxOnlyGrad,
  testAbsForward,
  testAbsGrad,
  testMseLossForward,
  testMseLossForwardValue,
  testMseLossGrad,
  testMseLossGradNotZero,
  testMseLossGradValue,
  testCrossEntropyForwardNonneg,
  testCrossEntropyGrad,
  testLinearForwardShape,
  testLinearBiasForward,
  testLinearGrad,
  testEmbeddingForwardShape,
  testEmbeddingForwardValues,
  testEmbeddingGrad,
  testEmbeddingDuplicateIndicesGrad,
  testRmsnormOutputShape,
  testRmsnormRmsEqualsOne,
  testRmsnormGrad,
  testRmsnorm3dGrad,
  testSequentialForward,
  testReluModule,
  testTanhModule,
  testLinearParameters,
  testLinearBiasParameters,
  testSequentialParameters,
  testRmsnormScaleNotInParams,
  testAdamStepDecreasesLoss,
  testAdamZeroGrad,
  testAdamConvergesLinearRegression,
  testZeroGradResets,
  testGradAccumulates,
  testChainRule,
  testDeepChain,
];

console.log('\n' + '='.repeat(60));
console.log(`  nn comprehensive test suite  (${tests.length} tests)`);
console.log('='.repeat(60));

let passed = 0;
let failed = 0;

for (const t of tests) {
  try {
    t();
    passed++;
    if (!t.name.startsWith('grad_check')) {
      console.log(`  [PASS] ${t.name}`);
    }
  } catch (e: any) {
    console.log(`  [FAIL] ${t.name}: ${e.message}`);
    failed++;
  }
}

console.log('-'.repeat(60));
console.log(`  ${passed}/${passed + failed} passed`);
if (failed > 0) console.log(`  ${failed} failed`);
console.log('='.repeat(60) + '\n');

if (failed > 0) process.exit(1);
