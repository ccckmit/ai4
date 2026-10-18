"""使用 NumPy tensor 版 nn 訓練字元級 GPT；請由範例庫載入。"""

import asyncio
import os
import random
import numpy as np
import nn

random.seed(42)
if not os.path.exists('input.txt'):
    raise FileNotFoundError('找不到 input.txt；請從範例庫載入 nn-gpt.py。')

docs = [line.strip() for line in open('input.txt', encoding='utf-8') if line.strip()]
chars = sorted(set(''.join(docs)))
token_of = {char: index for index, char in enumerate(chars)}
BOS = len(chars)
vocab_size, block_size = BOS + 1, 16

model = nn.GPT(vocab_size, block_size, width=24)
optimizer = nn.Adam(model.parameters)
num_parameters = sum(parameter.size for parameter in model.parameters.values())
print(f'訓練資料：{len(docs)} 行，詞彙表：{vocab_size} 個字元，模型參數：{num_parameters}')

num_steps = 1500
for step in range(num_steps):
    doc = docs[step % len(docs)]
    sequence = [BOS] + [token_of[char] for char in doc] + [BOS]
    tokens = np.array(sequence[:-1][:block_size])
    targets = np.array(sequence[1:][:block_size])
    loss, gradients = model.loss_and_gradients(tokens, targets)
    optimizer.step(gradients, 0.02 * (1 - step / num_steps) + 0.001)

    if step == 0 or (step + 1) % 150 == 0:
        print(f'步驟 {step + 1:4d} / {num_steps}，損失 {loss:.4f}')
    #if step % 25 == 0:
    #    await asyncio.sleep(0)  # 瀏覽器中讓出控制權，保持介面可更新。

print('\n--- 生成的句子（NumPy tensor 版） ---')
for sample_index in range(12):
    token_ids, sample = [BOS], []
    for _ in range(block_size):
        logits = model.forward(token_ids)[-1] / 0.6
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        token_id = random.choices(range(vocab_size), weights=probabilities)[0]
        if token_id == BOS:
            break
        sample.append(chars[token_id])
        token_ids.append(token_id)
    print(f'{sample_index + 1:2d}. {"".join(sample)}')
