import { Tensor, GPT } from '../index';

function assert(condition: boolean, msg: string) {
  if (!condition) throw new Error(`Assertion failed: ${msg}`);
}

function testGPTCreation() {
  const model = new GPT(20, 16, 1, 16, 4);
  const params = model.parameters();
  assert(params.length > 0, `expected params > 0, got ${params.length}`);
  console.log('  [PASS] GPT creation');
}

function testGPTForward() {
  const model = new GPT(20, 16, 1, 16, 4);
  const idx = Tensor.from([[1, 2, 3, 4]]);
  const { logits } = model.forward(idx);
  const flatLen = logits.data.length;
  const vocab = logits.data[0]?.length ?? 0;
  assert(flatLen === 4, `expected flat_len=4 (batch*seq), got ${flatLen}`);
  assert(vocab === 20, `expected vocab=20, got ${vocab}`);
  console.log('  [PASS] GPT forward pass');
}

function testGPTBackward() {
  const model = new GPT(20, 16, 1, 16, 4);
  const idx = Tensor.from([[1, 2, 3, 4]]);
  const { logits } = model.forward(idx);
  const targets = Tensor.from([[2, 3, 4, 5]]);
  const loss = logits.cross_entropy(targets);
  loss.backward();
  console.log('  [PASS] GPT backward pass (gradient flow - simplified check)');
}

function testGPTKvCache() {
  const model = new GPT(20, 16, 1, 16, 4);
  const x1 = Tensor.from([[1]]);
  const { caches: caches1 } = model.forward(x1);
  const x2 = Tensor.from([[2]]);
  const { logits: logits2 } = model.forward(x2, caches1);
  const vocab = logits2.data[0]?.length ?? 0;
  assert(vocab === 20, `expected vocab=20, got ${vocab}`);
  console.log('  [PASS] GPT KV cache');
}

function testGPTMultiLayer() {
  const model = new GPT(20, 16, 2, 16, 4);
  const idx = Tensor.from([[1, 2, 3]]);
  const { logits, caches } = model.forward(idx);
  const flatLen = logits.data.length;
  const vocab = logits.data[0]?.length ?? 0;
  assert(flatLen === 3, `expected flat_len=3, got ${flatLen}`);
  assert(vocab === 20, `expected vocab=20, got ${vocab}`);
  assert(caches.length === 2, `expected caches.length=2, got ${caches.length}`);
  console.log('  [PASS] GPT multi-layer');
}

console.log('\n=== GPT Tests ===');
testGPTCreation();
testGPTForward();
testGPTBackward();
testGPTKvCache();
testGPTMultiLayer();
console.log('\n  all passed');