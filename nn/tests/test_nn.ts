import { Tensor, Module, Linear, Embedding, RMSNorm, Adam } from '../index';

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(`Assertion failed: ${msg}`);
}

function testModuleParameters() {
  class DummyModule extends Module {
    p1: Tensor;
    p2: Tensor;
    constructor() {
      super();
      this.p1 = Tensor.from([[1]], true);
      this.p2 = Tensor.from([[2]], true);
    }
  }
  const m = new DummyModule();
  const params = m.parameters();
  assert(params.length === 2, `expected 2 params, got ${params.length}`);
  console.log('  [PASS] Module.parameters()');
}

function testLinearNoBias() {
  const linear = new Linear(3, 4);
  const x = Tensor.from([[1, 2, 3]], true);
  const y = linear.forward(x);
  assert(y.shape[0] === 1, `expected batch=1, got ${y.shape[0]}`);
  assert(y.shape[1] === 4, `expected out_features=4, got ${y.shape[1]}`);
  const params = linear.parameters();
  assert(params.length === 1, `expected 1 param, got ${params.length}`);
  console.log('  [PASS] Linear (no bias)');
}

function testLinearWithBias() {
  const linear = new Linear(3, 4, true);
  const x = Tensor.from([[1, 2, 3]], true);
  const y = linear.forward(x);
  assert(y.shape[0] === 1, `expected batch=1, got ${y.shape[0]}`);
  const params = linear.parameters();
  assert(params.length === 2, `expected 2 params, got ${params.length}`);
  console.log('  [PASS] Linear (with bias)');
}

function testEmbedding() {
  const embed = new Embedding(10, 4);
  const indices = Tensor.from([[1, 3, 5, 3, 1]]);
  const out = embed.forward(indices);
  assert(out.shape[0] === 5, `expected seq_len=5, got ${out.shape[0]}`);
  assert(out.shape[1] === 4, `expected embedding_dim=4, got ${out.shape[1]}`);
  console.log('  [PASS] Embedding');
}

function testEmbeddingBackward() {
  const embed = new Embedding(10, 4);
  const indices = Tensor.from([[1, 3, 5]]);
  const out = embed.forward(indices);
  const loss = out.sum();
  loss.backward();
  assert(embed.weight.grad !== undefined, 'expected weight.grad to be defined');
  console.log('  [PASS] Embedding backward');
}

function testRMSNorm() {
  const norm = new RMSNorm(4);
  const x = Tensor.from([[1, 2, 3, 4]], true);
  const y = norm.forward(x);
  assert(y.shape[0] === 1, `expected batch=1, got ${y.shape[0]}`);
  assert(y.shape[1] === 4, `expected dim=4, got ${y.shape[1]}`);
  console.log('  [PASS] RMSNorm');
}

function testRMSNormBackward() {
  const norm = new RMSNorm(4);
  const x = Tensor.from([[1, 2, 3, 4]], true);
  const y = norm.forward(x);
  const loss = y.sum();
  loss.backward();
  console.log('  [PASS] RMSNorm backward');
}

function testAdamOptimizer() {
  const p = Tensor.from([[1, 2, 3]], true);
  p.grad = [[0.1, 0.2, 0.3]];
  const optim = new Adam([p], 0.01);
  const oldData = p.data[0].slice();
  optim.step();
  const newData = p.data[0];
  const dataChanged = oldData.some((v, i) => v !== newData[i]);
  assert(dataChanged, 'expected data to change after step');
  optim.zeroGrad();
  const gradZero = p.grad[0].every(v => v === 0);
  assert(gradZero, 'expected grad to be zero after zeroGrad');
  console.log('  [PASS] Adam optimizer');
}

function testLinearBackward() {
  const linear = new Linear(2, 1);
  const x = Tensor.from([[1, 2]], true);
  const y = linear.forward(x);
  const loss = y.sum();
  loss.backward();
  assert(linear.weight.grad !== undefined, 'expected weight.grad to be defined');
  console.log('  [PASS] Linear backward');
}

console.log('\n=== nn Module Tests ===');
testModuleParameters();
testLinearNoBias();
testLinearWithBias();
testEmbedding();
testEmbeddingBackward();
testRMSNorm();
testRMSNormBackward();
testAdamOptimizer();
testLinearBackward();
console.log('\n  all passed');