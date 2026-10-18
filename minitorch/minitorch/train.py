"""訓練流程：分類與語言模型的訓練迴圈、取樣，以及可直接呼叫的完整 demo。"""

import random
import time
import numpy as np
from . import functional as F
from .data import DataLoader, TensorDataset, pad_collate
from .optim import AdamW, LambdaLR, clip_grad_norm_, warmup_cosine
from .tensor import no_grad


def _fit(model, loader, loss_fn, optimizer, scheduler, epochs, clip_norm, log_every, after_epoch=None):
    total, step, start = len(loader) * epochs, 0, time.time()
    for epoch in range(epochs):
        model.train()
        for batch in loader:
            loss = loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            if clip_norm:
                clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            step += 1
            if step == 1 or step % log_every == 0:
                print(f'步驟 {step:5d} / {total}，損失 {loss.item():.4f}，{time.time() - start:.0f} 秒')
        if after_epoch:
            after_epoch(epoch)


def fit_classifier(model, loader, test_data, optimizer, scheduler=None, epochs=1, clip_norm=None, log_every=100):
    x_test, y_test = test_data
    _fit(model, loader, lambda m, batch: F.cross_entropy(m(batch[0]), batch[1]), optimizer, scheduler,
         epochs, clip_norm, log_every,
         after_epoch=lambda e: print(f'第 {e + 1} 輪結束：測試準確率 {model.accuracy(x_test, y_test):.2%}'))


def fit_language_model(model, loader, optimizer, scheduler=None, epochs=1, clip_norm=None, log_every=100):
    _fit(model, loader, lambda m, batch: F.cross_entropy(m(batch[0]), batch[1], ignore_index=-100),
         optimizer, scheduler, epochs, clip_norm, log_every)


def generate(model, corpus, num_samples=12, temperature=0.6):
    """從 BOS 開始逐字取樣，遇到 BOS 或達到 block_size 就結束。"""
    model.eval()
    sentences = []
    with no_grad():
        for _ in range(num_samples):
            token_ids, sample = [corpus.BOS], []
            for _ in range(corpus.block_size):
                logits = model(token_ids).data[-1] / temperature
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
                token_id = random.choices(range(corpus.vocab_size), weights=probabilities)[0]
                if token_id == corpus.BOS:
                    break
                sample.append(corpus.chars[token_id])
                token_ids.append(token_id)
            sentences.append(''.join(sample))
    return sentences


def run_classifier_demo(model, data, batch_size=64, epochs=2, lr=0.003, weight_decay=0.0,
                        clip_norm=None, log_every=100, seed=42):
    x_train, y_train, x_test, y_test = data
    print(f'訓練資料：{len(x_train)} 張，測試資料：{len(x_test)} 張，模型參數：{model.num_parameters()}')
    loader = DataLoader(TensorDataset(x_train, y_train), batch_size, shuffle=True, drop_last=True, seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    total = len(loader) * epochs
    scheduler = LambdaLR(optimizer, warmup_cosine(total, warmup_steps=min(50, total // 10), min_ratio=0.1))
    fit_classifier(model, loader, (x_test, y_test), optimizer, scheduler, epochs, clip_norm, log_every)
    print('\n--- 前 10 張測試圖的預測 ---')
    print('預測：', model.predict(x_test[:10]).tolist())
    print('答案：', y_test[:10].tolist())


def run_language_model_demo(model, corpus, title, batch_size=16, epochs=60, lr=0.01,
                            weight_decay=0.01, clip_norm=1.0, log_every=100, seed=42):
    examples = corpus.examples()
    print(f'{corpus.describe()}，模型參數：{model.num_parameters()}')
    loader = DataLoader(examples, batch_size, shuffle=True, collate_fn=pad_collate((0, -100)), seed=seed)
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.99), weight_decay=weight_decay)
    total = len(loader) * epochs
    scheduler = LambdaLR(optimizer, warmup_cosine(total, warmup_steps=total // 20, min_ratio=0.05))
    fit_language_model(model, loader, optimizer, scheduler, epochs, clip_norm, log_every)
    print(f'\n--- 生成的句子（{title}） ---')
    for index, sentence in enumerate(generate(model, corpus), 1):
        print(f'{index:2d}. {sentence}')
