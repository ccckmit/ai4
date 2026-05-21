import { Tensor, cat } from './tensor';
import { Module, Linear, Embedding, RMSNorm } from './nn';

export class CausalSelfAttention extends Module {
  wq: Linear;
  wk: Linear;
  wv: Linear;
  wo: Linear;
  n_head: number;
  head_dim: number;

  constructor(n_embd: number, n_head: number) {
    super();
    this.n_head = n_head;
    this.head_dim = n_embd / n_head;
    this.wq = new Linear(n_embd, n_embd, false);
    this.wk = new Linear(n_embd, n_embd, false);
    this.wv = new Linear(n_embd, n_embd, false);
    this.wo = new Linear(n_embd, n_embd, false);
  }

  forward(x: Tensor, kv_cache?: { k: Tensor; v: Tensor }): { out: Tensor; cache: { k: Tensor; v: Tensor } } {
    const B = x.shape[0];
    const T = x.shape[1];
    const C = x.shape[2];

    let q = this.wq.forward(x);
    let k = this.wk.forward(x);
    let v = this.wv.forward(x);

    const nh = this.n_head;
    const hd = this.head_dim;
    q = q.reshape(B, T, nh, hd).transpose(1, 2);
    k = k.reshape(B, T, nh, hd).transpose(1, 2);
    v = v.reshape(B, T, nh, hd).transpose(1, 2);

    if (kv_cache) {
      k = cat([kv_cache.k, k], 2);
      v = cat([kv_cache.v, v], 2);
    }

    const T_k = k.shape[2];
    const scale = 1.0 / Math.sqrt(hd);
    let attn = q.matmul(k.transpose(2, 3)).mul(scale);

    if (T > 1) {
      const mask_data: number[] = [];
      for (let i = 0; i < T; i++) {
        for (let j = 0; j < T_k; j++) {
          mask_data.push(j > i ? -Infinity : 0);
        }
      }
      const mask = new Tensor(mask_data, [1, 1, T, T_k], false);
      attn = attn.add(mask);
    }

    const attn_weights = attn.softmax(3);
    const out = attn_weights.matmul(v);
    const out_reshaped = out.transpose(1, 2).reshape(B, T, C);
    return { out: this.wo.forward(out_reshaped), cache: { k, v } };
  }

  __call__(x: Tensor, kv_cache?: { k: Tensor; v: Tensor }): { out: Tensor; cache: { k: Tensor; v: Tensor } } {
    return this.forward(x, kv_cache);
  }
}

function relu(x: Tensor): Tensor {
  return x.relu();
}

export class TransformerBlock extends Module {
  attn: CausalSelfAttention;
  mlp_fc1: Linear;
  mlp_fc2: Linear;
  ln1: RMSNorm;
  ln2: RMSNorm;

  constructor(n_embd: number, n_head: number) {
    super();
    this.attn = new CausalSelfAttention(n_embd, n_head);
    this.mlp_fc1 = new Linear(n_embd, 4 * n_embd, false);
    this.mlp_fc2 = new Linear(4 * n_embd, n_embd, false);
    this.ln1 = new RMSNorm(n_embd);
    this.ln2 = new RMSNorm(n_embd);
  }

  forward(x: Tensor, kv_cache?: { k: Tensor; v: Tensor }): { out: Tensor; cache: { k: Tensor; v: Tensor } } {
    const a = this.attn.forward(this.ln1.forward(x), kv_cache);
    const h = x.add(a.out);
    const h2 = this.ln2.forward(h);
    const mlp_out = this.mlp_fc2.forward(relu(this.mlp_fc1.forward(h2)));
    return { out: h.add(mlp_out), cache: a.cache };
  }
}

export class GPT extends Module {
  tok_emb: Embedding;
  pos_emb: Embedding;
  blocks: TransformerBlock[];
  ln_f: RMSNorm;
  head: Linear;
  block_size: number;

  constructor(vocab_size: number, block_size: number, n_layer: number, n_embd: number, n_head: number) {
    super();
    this.block_size = block_size;
    this.tok_emb = new Embedding(vocab_size, n_embd);
    this.pos_emb = new Embedding(block_size, n_embd);
    this.blocks = [];
    for (let i = 0; i < n_layer; i++) {
      this.blocks.push(new TransformerBlock(n_embd, n_head));
    }
    this.ln_f = new RMSNorm(n_embd);
    this.head = new Linear(n_embd, vocab_size, false);
  }

  forward(token_ids: Tensor, _kv_caches?: { k: Tensor; v: Tensor }[]): { logits: Tensor; caches: { k: Tensor; v: Tensor }[] } {
    const B = token_ids.shape[0];
    const T = token_ids.shape[1];
    const past_len = (_kv_caches && _kv_caches[0]) ? _kv_caches[0].k.shape[3] : 0;

    const pos_data: number[] = [];
    for (let t = 0; t < T; t++) {
      pos_data.push(past_len + t);
    }
    const pos = Tensor.from([pos_data], false);

    const tok_emb = this.tok_emb.forward(token_ids);
    const pos_emb = this.pos_emb.forward(pos);
    let h = tok_emb.add(pos_emb);

    const caches: { k: Tensor; v: Tensor }[] = [];
    for (let i = 0; i < this.blocks.length; i++) {
      const layer_cache = _kv_caches ? _kv_caches[i] : undefined;
      const result = this.blocks[i].forward(h, layer_cache);
      h = result.out;
      caches.push(result.cache);
    }

    h = this.ln_f.forward(h);
    const logits = this.head.forward(h);
    return { logits, caches };
  }
}
