from __future__ import annotations

import numpy as np
from .tensor import Tensor
from .nn import Adam, Module
from .gpt import GPT


def train_model(
    model: GPT,
    optimizer: Adam,
    docs: list[str],
    uchars: list[str],
    BOS: int,
    block_size: int,
    num_steps: int = 1000,
) -> GPT:
    """Train GPT model without KV Cache."""
    params = model.parameters()

    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        x = np.array([tokens[:n]], dtype=int)
        y = np.array([tokens[1:n+1]], dtype=int)

        optimizer.zero_grad()
        logits, _ = model(x, kv_caches=None)

        loss = logits.cross_entropy(y)
        loss.backward()

        max_norm = 1.0
        total_norm = np.sqrt(sum(np.sum(p.grad ** 2) for p in params))
        if total_norm > max_norm:
            clip_coef = max_norm / (total_norm + 1e-6)
            for p in params:
                p.grad *= clip_coef

        optimizer.step()
        optimizer.lr = 0.01 * (1 - step / num_steps)

        print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

    print()
    return model


def generate_samples(
    model: GPT,
    uchars: list[str],
    BOS: int,
    vocab_size: int,
    block_size: int,
    num_samples: int = 20,
    temperature: float = 0.5,
) -> list[str]:
    """Generate samples using KV Cache for autoregressive generation."""
    print("\n--- inference (new, hallucinated names) ---")
    results: list[str] = []

    for sample_idx in range(num_samples):
        current_token = BOS
        sample: list[str] = []
        kv_caches = None

        for pos_id in range(block_size):
            x = np.array([[current_token]], dtype=int)
            logits, kv_caches = model(x, kv_caches)

            last_logits = logits.data[0, 0, :]

            exps = np.exp(last_logits / temperature - np.max(last_logits / temperature))
            probs = exps / np.sum(exps)

            current_token = np.random.choice(range(vocab_size), p=probs)
            if current_token == BOS:
                break
            sample.append(uchars[current_token])

        generated_name = ''.join(sample)
        results.append(generated_name)
        print(f"sample {sample_idx+1:2d}: {generated_name}")

    return results