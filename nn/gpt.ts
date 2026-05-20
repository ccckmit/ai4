import { Tensor } from './tensor';
import { Module, Linear, Embedding, RMSNorm } from './optim';

export class CausalSelfAttention extends Module {
  q: Linear;
  k: Linear;
  v: Linear;
  proj: Linear;
  head_dim: number;
  n_head: number;

  constructor(n_embd: number, n_head: number) {
    super();
    this.n_head = n_head;
    this.head_dim = n_embd / n_head;
    this.q = new Linear(n_embd, n_embd, false);
    this.k = new Linear(n_embd, n_embd, false);
    this.v = new Linear(n_embd, n_embd, false);
    this.proj = new Linear(n_embd, n_embd, false);
  }

  forward(x: Tensor, _kv_caches?: { k: Tensor; v: Tensor }[]): { out: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    const B = x.data.length;
    const seq_len = x.data[0]?.length ?? 0;
    const C = this.n_head * this.head_dim;

    const q: Tensor = this.q.forward(x);
    const k: Tensor = this.k.forward(x);
    const v: Tensor = this.v.forward(x);

    const q_data: number[][] = [];
    const k_data: number[][] = [];
    const v_data: number[][] = [];

    for (let b = 0; b < B; b++) {
      for (let t = 0; t < seq_len; t++) {
        const q_row = (q.data[b] ?? []).slice(t * this.n_head, (t + 1) * this.n_head);
        const k_row = (k.data[b] ?? []).slice(t * this.n_head, (t + 1) * this.n_head);
        const v_row = (v.data[b] ?? []).slice(t * this.n_head, (t + 1) * this.n_head);
        q_data.push(q_row);
        k_data.push(k_row);
        v_data.push(v_row);
      }
    }

    const scale = 1.0 / Math.sqrt(this.head_dim);
    const out_data: number[][] = [];

    for (let b = 0; b < B; b++) {
      for (let t = 0; t < seq_len; t++) {
        const q_idx = b * seq_len + t;
        const q_t: number[] = q_data[q_idx] ?? [];
        let score = 0;
        for (let tt = 0; tt <= t; tt++) {
          const k_idx = b * seq_len + tt;
          const k_t: number[] = k_data[k_idx] ?? [];
          const v_t: number[] = v_data[k_idx] ?? [];
          let dot = 0;
          for (let h = 0; h < this.n_head; h++) {
            dot += (q_t[h] ?? 0) * (k_t[h] ?? 0);
          }
          dot *= scale;
          let v_sum = 0;
          for (let h = 0; h < this.n_head; h++) {
            v_sum += v_t[h] ?? 0;
          }
          score += dot * v_sum;
        }
        out_data.push([score]);
      }
    }

    const out = new Tensor(out_data, [q, k, v]);
    const proj_out = this.proj.forward(out);
    return { out: proj_out, caches: [{ k, v }] };
  }

  __call__(x: Tensor, kv_caches?: { k: Tensor; v: Tensor }[]): { out: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    return this.forward(x, kv_caches);
  }
}

export class TransformerBlock extends Module {
  attn: CausalSelfAttention;
  mlp: Linear[];
  norm1: RMSNorm;
  norm2: RMSNorm;

  constructor(n_embd: number, n_head: number) {
    super();
    this.attn = new CausalSelfAttention(n_embd, n_head);
    this.norm1 = new RMSNorm(n_embd);
    this.norm2 = new RMSNorm(n_embd);
    this.mlp = [
      new Linear(n_embd, 4 * n_embd, false),
      new Linear(4 * n_embd, n_embd, false),
    ];
  }

  forward(x: Tensor, _kv_caches?: { k: Tensor; v: Tensor }[]): { out: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    const xn = this.norm1.forward(x);
    const { out, caches } = this.attn.forward(xn, _kv_caches);
    const h = x.add(out);

    const hn = this.norm2.forward(h);
    const ff_data: number[][] = hn.data.map((row: number[]) => {
      const hidden: number[] = this.mlp[0].forward(Tensor.from([row])).data[0] ?? [];
      const gelu: number[] = hidden.map((v: number) =>
        0.5 * v * (1 + Math.tanh(0.797885 * v + 0.044715 * v * v * v))
      );
      return this.mlp[1].forward(Tensor.from([gelu])).data[0] ?? [];
    });
    const ff_out = new Tensor(ff_data, [hn]);
    return { out: h.add(ff_out), caches };
  }

  __call__(x: Tensor, kv_caches?: { k: Tensor; v: Tensor }[]): { out: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    return this.forward(x, kv_caches);
  }
}

export class GPT extends Module {
  tok_emb: Embedding;
  pos_emb: Embedding;
  blocks: TransformerBlock[];
  head: Linear;
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_embd: number;
  n_head: number;

  constructor(vocab_size: number, block_size: number, n_layer: number, n_embd: number, n_head: number) {
    super();
    this.vocab_size = vocab_size;
    this.block_size = block_size;
    this.n_layer = n_layer;
    this.n_embd = n_embd;
    this.n_head = n_head;
    this.tok_emb = new Embedding(vocab_size, n_embd);
    this.pos_emb = new Embedding(block_size, n_embd);
    this.blocks = Array.from({ length: n_layer }, () => new TransformerBlock(n_embd, n_head));
    this.head = new Linear(n_embd, vocab_size, false);
  }

  forward(token_ids: Tensor, _kv_caches?: { k: Tensor; v: Tensor }[]): { logits: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    const B = token_ids.data.length;
    const T = token_ids.data[0]?.length ?? 0;

    const pos_data: number[][] = [];
    for (let t = 0; t < T; t++) {
      const row: number[] = [];
      for (let i = 0; i < T; i++) {
        row.push(t);
      }
      pos_data.push(row);
    }
    const pos = Tensor.from(pos_data, false);

    const x = this.tok_emb.forward(token_ids);
    const px = this.pos_emb.forward(pos);
    let h = x.add(px);

    const all_caches: { k: Tensor; v: Tensor }[] = [];
    for (const block of this.blocks) {
      const { out, caches } = block.forward(h, _kv_caches);
      h = out;
      if (caches[0]) all_caches.push(caches[0]);
    }

    const logits = this.head.forward(h);
    return { logits, caches: all_caches };
  }

  __call__(x: Tensor, kv_caches?: { k: Tensor; v: Tensor }[]): { logits: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    return this.forward(x, kv_caches);
  }

  generate(token_ids: Tensor, max_new_tokens: number): Tensor[] {
    const result: Tensor[] = [token_ids];
    let input = token_ids;
    for (let _i = 0; _i < max_new_tokens; _i++) {
      const { logits } = this.forward(input);
      const last_row = logits.data[logits.data.length - 1] ?? [];
      const max_val = Math.max(...last_row);
      const exp_sum = last_row.reduce((s, v) => s + Math.exp(v - max_val), 0);
      const probs = last_row.map(v => Math.exp(v - max_val) / exp_sum);
      const next_idx = probs.reduce((m, p, i) => (p > m[1] ? [i, p] : m), [0, 0])[0];
      const next_tok = Tensor.from([[next_idx]]);
      result.push(next_tok);
      input = next_tok;
    }
    return result;
  }
}