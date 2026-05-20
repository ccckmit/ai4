"""
GPT language model implementation with KV Cache support.
CausalSelfAttention: multi-head self-attention with causal masking
MLP: feed-forward network
Block: single Transformer block
GPT: full language model
"""

from __future__ import annotations

import numpy as np
from .tensor import Tensor, cat
from .nn import Module, Linear, Embedding, RMSNorm


class CausalSelfAttention(Module):
    """
    Multi-head causal self-attention.
    Supports KV Cache for efficient autoregressive inference.

    Key insight: When kv_cache is provided, concatenate past K/V with current K/V
    before computing attention scores. This avoids recomputing attention over the
    entire sequence history for each new token.
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        self.n_head: int = n_head
        self.head_dim: int = n_embd // n_head

        self.wq = Linear(n_embd, n_embd)
        self.wk = Linear(n_embd, n_embd)
        self.wv = Linear(n_embd, n_embd)
        self.wo = Linear(n_embd, n_embd)

    def __call__(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        """
        Forward pass with optional KV cache.

        Args:
            x: input tensor of shape (B, T, C) where B=batch, T=seq_len, C=n_embd
            kv_cache: tuple (past_k, past_v) from previous forward passes

        Returns:
            output: transformed tensor (B, T, C)
            new_cache: (k, v) tensors including current step for next forward pass
        """
        B, T, C = x.data.shape

        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            past_k, past_v = kv_cache
            k = cat([past_k, k], axis=2)
            v = cat([past_v, v], axis=2)

        T_k = k.data.shape[2]

        attn_logits = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)

        if T > 1:
            mask = np.triu(np.ones((T, T_k)), k=1) == 1
            attn_logits = attn_logits.masked_fill(mask, float("-inf"))

        attn_weights = attn_logits.softmax(axis=-1)
        out = attn_weights @ v

        out = out.transpose(1, 2).reshape(B, T, C)

        return self.wo(out), (k, v)


class MLP(Module):
    """
    Feed-forward network within each Transformer block.
    Expands to 4x hidden dimension, applies GELU, then projects back.
    """

    def __init__(self, n_embd: int) -> None:
        self.fc1 = Linear(n_embd, 4 * n_embd)
        self.fc2 = Linear(4 * n_embd, n_embd)

    def __call__(self, x: Tensor) -> Tensor:
        return self.fc2(self.fc1(x).relu())


class Block(Module):
    """
    Single Transformer block with pre-normalization (Pre-LN).
    Structure: RMSNorm -> Attention -> residual
              -> RMSNorm -> MLP -> residual

    Pre-LN is more stable during training compared to original Post-LN.
    """

    def __init__(self, n_embd: int, n_head: int) -> None:
        self.attn = CausalSelfAttention(n_embd, n_head)
        self.mlp = MLP(n_embd)
        self.ln1 = RMSNorm(n_embd)
        self.ln2 = RMSNorm(n_embd)

    def __call__(
        self,
        x: Tensor,
        kv_cache: tuple[Tensor, Tensor] | None = None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """
        Args:
            x: input tensor (B, T, C)
            kv_cache: optional cache from previous forward passes

        Returns:
            output: transformed tensor (B, T, C)
            new_cache: updated KV cache for next forward pass
        """
        attn_out, new_cache = self.attn(self.ln1(x), kv_cache)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x, new_cache


class GPT(Module):
    """
    GPT language model with embedding, positional encoding, stacked Transformer blocks,
    final normalization, and language modeling head.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_layer: int = 1,
        n_embd: int = 16,
        n_head: int = 4,
    ) -> None:
        self.block_size: int = block_size
        self.wte = Embedding(vocab_size, n_embd)
        self.wpe = Embedding(block_size, n_embd)
        self.blocks: list[Block] = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = RMSNorm(n_embd)
        self.lm_head = Linear(n_embd, vocab_size)

    def __call__(
        self,
        idx: np.ndarray,
        kv_caches: list[tuple[Tensor, Tensor]] | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor] | None]]:
        """
        Forward pass.

        Args:
            idx: token indices of shape (B, T) where T is sequence length
            kv_caches: list of (k, v) tuples from previous forward passes

        Returns:
            logits: predicted next-token logits of shape (B, T, vocab_size)
            new_caches: updated list of KV caches for next forward pass
        """
        B, T = idx.shape

        past_len = kv_caches[0][0].data.shape[2] if kv_caches is not None else 0
        pos = np.arange(past_len, past_len + T, dtype=int)

        tok_emb = self.wte(idx)
        pos_emb = self.wpe(pos)
        x = tok_emb + pos_emb

        new_caches: list[tuple[Tensor, Tensor] | None] = []
        for i, block in enumerate(self.blocks):
            layer_cache = kv_caches[i] if kv_caches is not None else None
            x, new_cache = block(x, layer_cache)
            new_caches.append(new_cache)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        return logits, new_caches