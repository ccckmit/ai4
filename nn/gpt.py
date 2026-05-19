import numpy as np
from .tensor import Tensor, cat
from .optim import Module, Linear, Embedding, RMSNorm

"""
  因果自注意力（支援 KV Cache）。
  推論時使用 KV Cache 拼接過去與當前的 K/V，避免重複計算。
  訓練時 T>1 需使用因果遮蔽，推論時 T=1 不需遮蔽。
"""
class CausalSelfAttention(Module):
    def __init__(self, n_embd, n_head):
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        self.wq = Linear(n_embd, n_embd)
        self.wk = Linear(n_embd, n_embd)
        self.wv = Linear(n_embd, n_embd)
        self.wo = Linear(n_embd, n_embd)

    def __call__(self, x, kv_cache=None):
        B, T, C = x.data.shape
        
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 若有快取，將過去的 K/V 與當前的 K/V 沿序列維度拼接
        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = cat([past_k, k], axis=2)
            v = cat([past_v, v], axis=2)
            
        T_k = k.data.shape[2]

        attn_logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # 訓練時使用因果遮蔽；推論時 T=1，不需要遮蔽
        if T > 1:
            mask = np.triu(np.ones((T, T_k)), k=1) == 1
            attn_logits = attn_logits.masked_fill(mask, float('-inf'))
        
        attn_weights = attn_logits.softmax(axis=-1)
        out = attn_weights @ v 
        out = out.transpose(1, 2).reshape(B, T, C)
        
        return self.wo(out), (k, v)

class MLP(Module):
    def __init__(self, n_embd):
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.fc2 = Linear(4 * n_embd, n_embd)

    def __call__(self, x):
        return self.fc2(self.fc1(x).relu())

"""Transformer Block，返回更新後的 KV Cache"""
class Block(Module):
    def __init__(self, n_embd, n_head):
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = MLP(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def __call__(self, x, kv_cache=None):
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache

"""
  GPT 語言模型（含 KV Cache 支援）。
  推論時接收過去的 kv_caches，回傳新的 caches 供下一步使用。
"""
class GPT(Module):
    def __init__(self, vocab_size, block_size, n_layer=1, n_embd=16, n_head=4):
        self.block_size = block_size
        self.wte = Embedding(vocab_size, n_embd)
        self.wpe = Embedding(block_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size)

    def __call__(self, idx, kv_caches=None):
        B, T = idx.shape
        
        # 計算位置偏移：若有快取，位置要從 past_len 開始
        past_len = kv_caches[0][0].data.shape[2] if kv_caches is not None else 0
        pos = np.arange(past_len, past_len + T, dtype=int)
        
        tok_emb = self.wte(idx)       
        pos_emb = self.wpe(pos)       
        x = tok_emb + pos_emb
        
        new_caches = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits, new_caches