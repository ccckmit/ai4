import { Tensor } from './tensor';
import { GPT } from './gpt';
import { Adam } from './nn';

export function train_model(
  model: GPT,
  optimizer: Adam,
  docs: string[],
  uchars: string[],
  BOS: number,
  block_size: number,
  num_steps = 1000
): GPT {
  const params = model.parameters();

  for (let step = 0; step < num_steps; step++) {
    const doc = docs[step % docs.length];
    const tokens = [BOS, ...doc.split('').map(ch => uchars.indexOf(ch)), BOS];
    const n = Math.min(block_size, tokens.length - 1);

    const x = [tokens.slice(0, n)];
    const y = [tokens.slice(1, n + 1)];

    optimizer.zeroGrad();
    const { logits } = model.forward(Tensor.from(x));

    const loss = logits.cross_entropy(Tensor.from(y));
    loss.backward();

    let total_norm = 0;
    for (const p of params) {
      if (p.grad) {
        for (const row of p.grad) {
          for (const g of row) {
            total_norm += g * g;
          }
        }
      }
    }
    total_norm = Math.sqrt(total_norm);

    const max_norm = 1.0;
    if (total_norm > max_norm) {
      const clip_coef = max_norm / (total_norm + 1e-6);
      for (const p of params) {
        if (p.grad) {
          p.grad = p.grad.map((row: number[]) => row.map((g: number) => g * clip_coef));
        }
      }
    }

    optimizer.step();
    (optimizer as unknown as { lr: number }).lr = 0.01 * (1 - step / num_steps);

    console.log(`step ${String(step + 1).padStart(4)} / ${num_steps} | loss ${(loss.data[0]?.[0] ?? 0).toFixed(4)}`);
  }

  return model;
}

export function generate_samples(
  model: GPT,
  uchars: string[],
  BOS: number,
  vocab_size: number,
  block_size: number,
  num_samples = 20,
  temperature = 0.5
): string[] {
  console.log('\n--- inference (new, hallucinated names) ---');
  const results: string[] = [];

  for (let sample_idx = 0; sample_idx < num_samples; sample_idx++) {
    let current_token = BOS;
    const sample: string[] = [];
    let kv_caches: { k: Tensor; v: Tensor }[] | undefined;

    for (let pos_id = 0; pos_id < block_size; pos_id++) {
      const x = [[current_token]];
      const { logits, caches } = model.forward(Tensor.from(x), kv_caches);
      kv_caches = caches;

      const last_logits: number[] = logits.data[0] ?? [];
      const max_log = Math.max(...last_logits);
      const exps = last_logits.map((v: number) => Math.exp((v - max_log) / temperature));
      const sum_exp = exps.reduce((a: number, b: number) => a + b, 0);
      const probs = exps.map((v: number) => v / sum_exp);

      let cumsum = 0;
      const r = Math.random();
      let next_token = 0;
      for (let v = 0; v < vocab_size; v++) {
        cumsum += probs[v] ?? 0;
        if (r <= cumsum) {
          next_token = v;
          break;
        }
      }

      if (next_token === BOS) break;
      sample.push(uchars[next_token] ?? '');
      current_token = next_token;
    }

    const generated_name = sample.join('');
    results.push(generated_name);
    console.log(`sample ${String(sample_idx + 1).padStart(2)}: ${generated_name}`);
  }

  return results;
}