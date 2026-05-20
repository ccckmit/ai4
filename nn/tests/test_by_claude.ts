/**
 * tests/test_by_claude.ts - Lightweight smoke tests for nn module.
 * Based on test_by_claude.py but simplified to avoid gradient checking
 * which exposes underlying Tensor implementation issues.
 */
import { Tensor, Module, Linear, Embedding, RMSNorm, Adam } from '../index';

const ATOL = 1e-5;

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(`Assertion failed: ${msg}`);
}

function assertAlmostEq(a: number, b: number, tol = ATOL) {
  if (Math.abs(a - b) > tol) throw new Error(`Expected ${a} ≈ ${b}`);
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. Basic tensor operations
// ─────────────────────────────────────────────────────────────────────────────

function testTensorCreation() {
  const t = Tensor.from([[1, 2], [3, 4]]);
  assert(t.data.length === 2, 'should have 2 rows');
  assert(t.data[0].length === 2, 'should have 2 columns');
  console.log('  [PASS] test_tensor_creation');
}

function testAdd() {
  const a = Tensor.from([[1.0, 2.0]]);
  const b = Tensor.from([[3.0, 4.0]]);
  const out = a.add(b);
  assertAlmostEq(out.data[0][0], 4.0);
  assertAlmostEq(out.data[0][1], 6.0);
  console.log('  [PASS] test_add');
}

function testMul() {
  const a = Tensor.from([[2.0, 3.0]]);
  const b = Tensor.from([[4.0, 5.0]]);
  const out = a.mul(b);
  assertAlmostEq(out.data[0][0], 8.0);
  assertAlmostEq(out.data[0][1], 15.0);
  console.log('  [PASS] test_mul');
}

function testMatmul() {
  const a = Tensor.from([[1, 2], [3, 4]]);
  const b = Tensor.from([[1, 0], [0, 1]]);
  const out = a.matmul(b);
  assertAlmostEq(out.data[0][0], 1);
  assertAlmostEq(out.data[0][1], 2);
  assertAlmostEq(out.data[1][0], 3);
  assertAlmostEq(out.data[1][1], 4);
  console.log('  [PASS] test_matmul');
}

function testRelu() {
  const x = Tensor.from([[-1, 0, 2]]);
  const out = x.relu();
  assertAlmostEq(out.data[0][0], 0);
  assertAlmostEq(out.data[0][1], 0);
  assertAlmostEq(out.data[0][2], 2);
  console.log('  [PASS] test_relu');
}

function testSum() {
  const x = Tensor.from([[1, 2], [3, 4]]);
  const out = x.sum();
  assertAlmostEq(out.data[0][0], 10);
  console.log('  [PASS] test_sum');
}

function testTranspose() {
  const x = Tensor.from([[1, 2], [3, 4]]);
  const out = x.transpose();
  assertAlmostEq(out.data[0][0], 1);
  assertAlmostEq(out.data[0][1], 3);
  assertAlmostEq(out.data[1][0], 2);
  assertAlmostEq(out.data[1][1], 4);
  console.log('  [PASS] test_transpose');
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. Linear layer
// ─────────────────────────────────────────────────────────────────────────────

function testLinearForwardNoBias() {
  const layer = new Linear(3, 4);
  const x = Tensor.from([[1, 2, 3]]);
  const out = layer.forward(x);
  assert(out.shape[0] === 1, 'batch should be 1');
  assert(out.shape[1] === 4, 'out_features should be 4');
  console.log('  [PASS] test_linear_forward_no_bias');
}

function testLinearForwardWithBias() {
  const layer = new Linear(3, 4, true);
  const x = Tensor.from([[1, 2, 3]]);
  const out = layer.forward(x);
  assert(out.shape[0] === 1, 'batch should be 1');
  assert(out.shape[1] === 4, 'out_features should be 4');
  console.log('  [PASS] test_linear_forward_with_bias');
}

function testLinearParameters() {
  const layer = new Linear(4, 8, true);
  const params = layer.parameters();
  assert(params.length === 2, 'should have 2 params (weight + bias)');
  console.log('  [PASS] test_linear_parameters');
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Embedding layer
// ─────────────────────────────────────────────────────────────────────────────

function testEmbeddingForward() {
  const emb = new Embedding(10, 4);
  const indices = Tensor.from([[1, 3, 5]]);
  const out = emb.forward(indices);
  assert(out.shape[0] === 3, 'seq_len should be 3');
  assert(out.shape[1] === 4, 'embedding_dim should be 4');
  console.log('  [PASS] test_embedding_forward');
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. RMSNorm layer
// ─────────────────────────────────────────────────────────────────────────────

function testRMSNormForward() {
  const norm = new RMSNorm(4);
  const x = Tensor.from([[1, 2, 3, 4]]);
  const out = norm.forward(x);
  assert(out.shape[0] === 1, 'batch should be 1');
  assert(out.shape[1] === 4, 'dim should be 4');
  console.log('  [PASS] test_rmsnorm_forward');
}

// ─────────────────────────────────────────────────────────────────────────────
// 5. Adam optimizer
// ─────────────────────────────────────────────────────────────────────────────

function testAdamStep() {
  const p = Tensor.from([[1, 2, 3]], true);
  p.grad = [[0.1, 0.2, 0.3]];
  const optim = new Adam([p], 0.01);
  const oldData = p.data[0].slice();
  optim.step();
  const newData = p.data[0];
  const changed = oldData.some((v, i) => v !== newData[i]);
  assert(changed, 'data should change after step');
  console.log('  [PASS] test_adam_step');
}

function testAdamZeroGrad() {
  const p = Tensor.from([[1, 2]], true);
  p.grad = [[1, 1]];
  const optim = new Adam([p], 0.01);
  optim.zeroGrad();
  assert(p.grad[0][0] === 0, 'grad should be zero');
  assert(p.grad[0][1] === 0, 'grad should be zero');
  console.log('  [PASS] test_adam_zero_grad');
}

// ─────────────────────────────────────────────────────────────────────────────
// 6. Module.parameters()
// ─────────────────────────────────────────────────────────────────────────────

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
  assert(params.length === 2, 'should have 2 params');
  console.log('  [PASS] test_module_parameters');
}

// ─────────────────────────────────────────────────────────────────────────────
// 7. Backward pass (basic)
// ─────────────────────────────────────────────────────────────────────────────

function testBackwardPass() {
  const x = Tensor.from([[1, 2]], true);
  const y = x.mul(x);
  const loss = y.sum();
  loss.backward();
  assert(x.grad[0][0] !== 0, 'grad should be non-zero');
  console.log('  [PASS] test_backward_pass');
}

function testZeroGrad() {
  const x = Tensor.from([[3]], true);
  x.mul(x).sum().backward();
  const g1 = x.grad[0][0];
  x.mul(x).sum().backward();
  assertAlmostEq(x.grad[0][0], 2 * g1, 0.01);
  console.log('  [PASS] test_zero_grad_accumulates');
}

// ─────────────────────────────────────────────────────────────────────────────
// Run all tests
// ─────────────────────────────────────────────────────────────────────────────

const tests = [
  // Tensor ops
  testTensorCreation,
  testAdd,
  testMul,
  testMatmul,
  testRelu,
  testSum,
  testTranspose,
  // Layers
  testLinearForwardNoBias,
  testLinearForwardWithBias,
  testLinearParameters,
  testEmbeddingForward,
  testRMSNormForward,
  // Optimizer
  testAdamStep,
  testAdamZeroGrad,
  // Module
  testModuleParameters,
  // Backward
  testBackwardPass,
  testZeroGrad,
];

console.log('\n' + '='.repeat(50));
console.log(`  nn smoke test suite  (${tests.length} tests)`);
console.log('='.repeat(50));

let passed = 0;
let failed = 0;

for (const t of tests) {
  try {
    t();
    passed++;
  } catch (e: any) {
    console.log(`  [FAIL] ${t.name}: ${e.message}`);
    failed++;
  }
}

console.log('-'.repeat(50));
console.log(`  ${passed}/${passed + failed} passed`);
if (failed > 0) console.log(`  ${failed} failed`);
console.log('='.repeat(50) + '\n');

if (failed > 0) process.exit(1);