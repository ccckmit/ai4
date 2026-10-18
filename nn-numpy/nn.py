"""NumPy tensor 版的極簡 GPT：手寫矩陣反向傳播，不建立純量運算圖。"""

import numpy as np


def _softmax(values):
    shifted = values - values.max(axis=-1, keepdims=True)
    exps = np.exp(shifted)
    return exps / exps.sum(axis=-1, keepdims=True)


class Adam:
    def __init__(self, parameters, beta1=0.9, beta2=0.99, epsilon=1e-8):
        self.parameters = parameters
        self.beta1, self.beta2, self.epsilon = beta1, beta2, epsilon
        self.first = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.second = {name: np.zeros_like(value) for name, value in parameters.items()}
        self.steps = 0

    def step(self, gradients, learning_rate):
        self.steps += 1
        for name, parameter in self.parameters.items():
            self.first[name] = self.beta1 * self.first[name] + (1 - self.beta1) * gradients[name]
            self.second[name] = self.beta2 * self.second[name] + (1 - self.beta2) * gradients[name] ** 2
            first = self.first[name] / (1 - self.beta1 ** self.steps)
            second = self.second[name] / (1 - self.beta2 ** self.steps)
            parameter -= learning_rate * first / (np.sqrt(second) + self.epsilon)


class GPT:
    """單層、單頭的 causal Transformer，所有權重梯度皆以 tensor 計算。"""

    def __init__(self, vocab_size, block_size, width=24, seed=42):
        self.block_size = block_size
        self.width = width
        rng = np.random.default_rng(seed)
        normal = lambda *shape: rng.normal(0, 0.08, shape)
        self.parameters = {
            'wte': normal(vocab_size, width),
            'wpe': normal(block_size, width),
            'wq': normal(width, width),
            'wk': normal(width, width),
            'wv': normal(width, width),
            'wo': normal(width, width),
            'w1': normal(width, 4 * width),
            'w2': normal(4 * width, width),
            'lm_head': normal(width, vocab_size),
        }

    def forward(self, token_ids, keep_cache=False):
        """輸入 shape (T,)，輸出每個位置的下一個字元 logits，shape (T, V)。"""
        p = self.parameters
        token_ids = np.asarray(token_ids, dtype=np.int64)
        length = len(token_ids)
        if length > self.block_size:
            raise ValueError(f'序列長度不可超過 block_size={self.block_size}')

        x = p['wte'][token_ids] + p['wpe'][:length]
        q, k, v = x @ p['wq'], x @ p['wk'], x @ p['wv']
        scores = q @ k.T / np.sqrt(self.width)
        future = np.triu(np.ones((length, length), dtype=bool), k=1)
        scores[future] = -1e9
        attention = _softmax(scores)
        attended = attention @ v
        x2 = x + attended @ p['wo']
        pre_activation = x2 @ p['w1']
        hidden = np.maximum(pre_activation, 0)
        x3 = x2 + hidden @ p['w2']
        logits = x3 @ p['lm_head']

        if not keep_cache:
            return logits
        return logits, (token_ids, x, q, k, v, future, attention, attended,
                        x2, pre_activation, hidden, x3)

    def loss_and_gradients(self, token_ids, targets):
        """交叉熵損失與全部參數的解析梯度。"""
        p = self.parameters
        logits, cache = self.forward(token_ids, keep_cache=True)
        token_ids, x, q, k, v, future, attention, attended, x2, pre, hidden, x3 = cache
        targets = np.asarray(targets, dtype=np.int64)
        length = len(token_ids)

        probabilities = _softmax(logits)
        loss = -np.log(probabilities[np.arange(length), targets] + 1e-12).mean()
        d_logits = probabilities.copy()
        d_logits[np.arange(length), targets] -= 1
        d_logits /= length
        gradients = {name: np.zeros_like(value) for name, value in p.items()}

        gradients['lm_head'] = x3.T @ d_logits
        d_x3 = d_logits @ p['lm_head'].T

        d_x2 = d_x3.copy()  # MLP 殘差連接
        gradients['w2'] = hidden.T @ d_x3
        d_hidden = d_x3 @ p['w2'].T
        d_pre = d_hidden * (pre > 0)
        gradients['w1'] = x2.T @ d_pre
        d_x2 += d_pre @ p['w1'].T

        d_x = d_x2.copy()  # attention 殘差連接
        gradients['wo'] = attended.T @ d_x2
        d_attended = d_x2 @ p['wo'].T
        d_attention = d_attended @ v.T
        d_v = attention.T @ d_attended
        # softmax Jacobian-vector product，逐列計算即可。
        d_scores = attention * (d_attention - (d_attention * attention).sum(axis=1, keepdims=True))
        d_scores[future] = 0
        scale = 1 / np.sqrt(self.width)
        d_q = d_scores @ k * scale
        d_k = d_scores.T @ q * scale
        gradients['wq'] = x.T @ d_q
        gradients['wk'] = x.T @ d_k
        gradients['wv'] = x.T @ d_v
        d_x += d_q @ p['wq'].T + d_k @ p['wk'].T + d_v @ p['wv'].T

        np.add.at(gradients['wte'], token_ids, d_x)
        gradients['wpe'][:length] += d_x
        return float(loss), gradients
