// ============================================================
//  gpt.rs  –  Causal Self-Attention, TransformerBlock, GPT
// ============================================================
use super::tensor::{cat, SimpleRng, Tensor};
use super::nn::{Embedding, Linear, RMSNorm};

pub struct CausalSelfAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_head: usize,
    pub head_dim: usize,
}

impl CausalSelfAttention {
    pub fn new(n_embd: usize, n_head: usize, rng: &mut SimpleRng) -> Self {
        let hd = n_embd / n_head;
        Self {
            wq: Linear::new(n_embd, n_embd, false, rng),
            wk: Linear::new(n_embd, n_embd, false, rng),
            wv: Linear::new(n_embd, n_embd, false, rng),
            wo: Linear::new(n_embd, n_embd, false, rng),
            n_head: n_head,
            head_dim: hd,
        }
    }

    /// Returns (output, (k_cache, v_cache))
    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<(Tensor, Tensor)>,
    ) -> (Tensor, (Tensor, Tensor)) {
        let xs = x.shape(); // [B, T, C]
        let (b, t, c) = (xs[0], xs[1], xs[2]);
        let nh = self.n_head;
        let hd = self.head_dim;

        // project
        let q = self.wq.forward(x).reshape(vec![b, t, nh, hd]).transpose(1, 2); // [B,nh,T,hd]
        let k_new = self.wk.forward(x).reshape(vec![b, t, nh, hd]).transpose(1, 2);
        let v_new = self.wv.forward(x).reshape(vec![b, t, nh, hd]).transpose(1, 2);

        // extend kv cache
        let (k, v) = if let Some((kc, vc)) = kv_cache {
            (cat(&[kc, k_new], 2), cat(&[vc, v_new], 2))
        } else {
            (k_new, v_new)
        };

        let t_k = k.shape()[2];
        let scale = 1.0 / (hd as f64).sqrt();
        let kt = k.transpose(2, 3); // [B,nh,hd,T_k]
        let mut attn = q.matmul(&kt).mul_scalar(scale); // [B,nh,T,T_k]

        // causal mask only when generating a full sequence (T > 1)
        if t > 1 {
            let mut mask_data = vec![0.0f64; t * t_k];
            for i in 0..t {
                for j in 0..t_k {
                    if j > i {
                        mask_data[i * t_k + j] = f64::NEG_INFINITY;
                    }
                }
            }
            let mask = Tensor::new(mask_data, vec![1, 1, t, t_k], false);
            attn = attn.add(&mask);
        }

        let aw = attn.softmax(3); // [B,nh,T,T_k]
        let o = aw.matmul(&v); // [B,nh,T,hd]
        let o2 = o.transpose(1, 2).reshape(vec![b, t, c]);
        (self.wo.forward(&o2), (k, v))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        [&self.wq, &self.wk, &self.wv, &self.wo]
            .iter()
            .flat_map(|l| l.parameters())
            .collect()
    }
}

pub struct TransformerBlock {
    pub attn: CausalSelfAttention,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
    pub ln1: RMSNorm,
    pub ln2: RMSNorm,
}

impl TransformerBlock {
    pub fn new(n_embd: usize, n_head: usize, rng: &mut SimpleRng) -> Self {
        Self {
            attn: CausalSelfAttention::new(n_embd, n_head, rng),
            mlp_fc1: Linear::new(n_embd, 4 * n_embd, false, rng),
            mlp_fc2: Linear::new(4 * n_embd, n_embd, false, rng),
            ln1: RMSNorm::new(n_embd),
            ln2: RMSNorm::new(n_embd),
        }
    }

    pub fn forward(
        &self,
        x: &Tensor,
        kv_cache: Option<(Tensor, Tensor)>,
    ) -> (Tensor, (Tensor, Tensor)) {
        let (a_out, cache) = self.attn.forward(&self.ln1.forward(x), kv_cache);
        let h = x.add(&a_out);
        let mlp_out = self.mlp_fc2.forward(&self.mlp_fc1.forward(&self.ln2.forward(&h)).relu());
        (h.add(&mlp_out), cache)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.attn.parameters();
        p.extend(self.mlp_fc1.parameters());
        p.extend(self.mlp_fc2.parameters());
        p.extend(self.ln1.parameters());
        p.extend(self.ln2.parameters());
        p
    }
}

pub struct GPT {
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: RMSNorm,
    pub head: Linear,
    pub block_size: usize,
}

impl GPT {
    pub fn new(
        vocab_size: usize,
        block_size: usize,
        n_layer: usize,
        n_embd: usize,
        n_head: usize,
        rng: &mut SimpleRng,
    ) -> Self {
        let mut blocks = Vec::new();
        for _ in 0..n_layer {
            blocks.push(TransformerBlock::new(n_embd, n_head, rng));
        }
        Self {
            tok_emb: Embedding::new(vocab_size, n_embd, rng),
            pos_emb: Embedding::new(block_size, n_embd, rng),
            blocks,
            ln_f: RMSNorm::new(n_embd),
            head: Linear::new(n_embd, vocab_size, false, rng),
            block_size,
        }
    }

    /// token_ids: [B, T]
    pub fn forward(
        &self,
        token_ids: &Tensor,
        kv_caches: Option<Vec<(Tensor, Tensor)>>,
    ) -> (Tensor, Vec<(Tensor, Tensor)>) {
        let ts = token_ids.shape();
        let (_, t) = (ts[0], ts[1]);
        let past_len = kv_caches
            .as_ref()
            .and_then(|c| c.first())
            .map(|(k, _)| k.shape()[3])   // [B, nh, T_past, hd] → dim 2
            .unwrap_or(0);

        // positional indices
        let pos_data: Vec<f64> = (0..t).map(|i| (past_len + i) as f64).collect();
        let pos = Tensor::new(pos_data, vec![1, t], false);

        let tok_e = self.tok_emb.forward(token_ids);
        let pos_e = self.pos_emb.forward(&pos);
        let mut h = tok_e.add(&pos_e);

        let mut new_caches: Vec<(Tensor, Tensor)> = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_cache = kv_caches.as_ref().and_then(|c| c.get(i)).cloned();
            let (h2, cache) = block.forward(&h, layer_cache);
            h = h2;
            new_caches.push(cache);
        }

        h = self.ln_f.forward(&h);
        let logits = self.head.forward(&h);
        (logits, new_caches)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut p = self.tok_emb.parameters();
        p.extend(self.pos_emb.parameters());
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.extend(self.ln_f.parameters());
        p.extend(self.head.parameters());
        p
    }
}
