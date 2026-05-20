import numpy as np
from .tensor import Tensor
from .nn import Adam
from .gpt import GPT

"""
  訓練 GPT 模型。
  訓練時不使用 KV Cache，直接計算整段序列的 loss。
  加入梯度裁剪（Gradient Clipping）防止梯度爆炸。
"""
def train_model(model, optimizer, docs, uchars, BOS, block_size, num_steps=1000):
    params = model.parameters()
    
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)
        
        x = np.array([tokens[:n]], dtype=int)
        y = np.array([tokens[1:n+1]], dtype=int)
        
        optimizer.zero_grad()
        logits, _ = model(x, kv_caches=None)  # 訓練不用 KV Cache
        
        loss = logits.cross_entropy(y)
        loss.backward()
        
        # 梯度裁剪：限制所有參數梯度的 L2 範數不超過 max_norm
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

"""
  推論函數：使用 KV Cache 加速自回歸生成。
  每次只輸入上一步產生的 token（長度 1），
  快取會自動累積歷史資訊。
"""
def generate_samples(model, uchars, BOS, vocab_size, block_size, num_samples=20, temperature=0.5):
    print("\n--- inference (new, hallucinated names) ---")
    results = []
    
    for sample_idx in range(num_samples):
        current_token = BOS
        sample = []
        kv_caches = None  # 每個新樣本清空快取
        
        for pos_id in range(block_size):
            x = np.array([[current_token]], dtype=int)
            logits, kv_caches = model(x, kv_caches)
            
            last_logits = logits.data[0, 0, :]  # 取當前 token 的 logits
            
            # Temperature Scaling + Softmax
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
