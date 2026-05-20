from __future__ import annotations

import os
import random
import numpy as np
from .nn import Adam
from .chargpt import generate_samples, train_model
from .gpt import GPT


def main() -> None:
    random.seed(42)
    np.random.seed(42)

    if not os.path.exists('data/input.txt'):
        import urllib.request
        names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
        urllib.request.urlretrieve(names_url, 'data/input.txt')

    docs = [line.strip() for line in open('data/input.txt') if line.strip()]
    random.shuffle(docs)
    print(f"num docs: {len(docs)}")

    uchars = sorted(set(''.join(docs)))
    BOS = len(uchars)
    vocab_size = len(uchars) + 1
    print(f"vocab size: {vocab_size}")

    block_size = 16
    model = GPT(vocab_size=vocab_size, block_size=block_size, n_layer=1, n_embd=16, n_head=4)
    print(f"num params: {len(model.parameters())}")

    optimizer = Adam(model.parameters(), lr=0.01)

    train_model(
        model=model, optimizer=optimizer, docs=docs, uchars=uchars,
        BOS=BOS, block_size=block_size, num_steps=1000
    )

    generate_samples(
        model=model, uchars=uchars, BOS=BOS, vocab_size=vocab_size,
        block_size=block_size, num_samples=20, temperature=0.5
    )


if __name__ == '__main__':
    main()