import { GPT, Tensor } from '../index';

describe('GPT', () => {
  test('creation has parameters', () => {
    const gpt = new GPT(20, 16, 1, 16, 4);
    const params = gpt.parameters();
    expect(params.length).toBeGreaterThan(0);
  });

  test('forward returns correct shape', () => {
    const gpt = new GPT(20, 16, 1, 16, 4);
    // GPT.forward expects a 2D tensor [batch, seq_len]
    const idx = Tensor.from([[1, 2, 3, 4]], false);
    const { logits, caches } = gpt.forward(idx, undefined);
    expect(logits.shape[0]).toBe(1);
    expect(logits.shape[1]).toBe(20);
  });
});