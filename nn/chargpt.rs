// ============================================================
//  chargpt.rs  –  training loop + generation
// ============================================================
use super::gpt::GPT;
use super::nn::Adam;
use super::tensor::Tensor;

pub fn train_model(
    model: &GPT,
    optimizer: &mut Adam,
    docs: &[String],
    uchars: &[char],
    bos: usize,
    block_size: usize,
    num_steps: usize,
) {
    for step in 0..num_steps {
        let doc = &docs[step % docs.len()];
        let mut tokens: Vec<usize> = vec![bos];
        for ch in doc.chars() {
            if let Some(idx) = uchars.iter().position(|&c| c == ch) {
                tokens.push(idx);
            }
        }
        tokens.push(bos);

        let n = (block_size).min(tokens.len() - 1);
        let x_data: Vec<f64> = tokens[..n].iter().map(|&t| t as f64).collect();
        let x = Tensor::new(x_data, vec![1, n], false);
        let targets: Vec<usize> = tokens[1..n + 1].to_vec();

        optimizer.zero_grad();
        let (logits, _) = model.forward(&x, None);

        // logits: [1, T, V] → flatten to [T, V]
        let ls = logits.shape();
        let (t, v) = (ls[1], ls[2]);
        let logits_flat = logits.reshape(vec![t, v]);
        let loss = logits_flat.cross_entropy(&targets);
        loss.backward();

        // gradient clipping
        let params = model.parameters();
        let total_norm: f64 = params
            .iter()
            .filter_map(|p| {
                if p.requires_grad() {
                    Some(p.grad().iter().map(|&g| g * g).sum::<f64>())
                } else {
                    None
                }
            })
            .sum::<f64>()
            .sqrt();

        let max_norm = 1.0f64;
        if total_norm > max_norm {
            let clip = max_norm / (total_norm + 1e-6);
            for p in &params {
                if p.requires_grad() {
                    let mut gm = p.grad_mut();
                    for g in gm.iter_mut() {
                        *g *= clip;
                    }
                }
            }
        }

        optimizer.step();
        optimizer.lr = 0.01 * (1.0 - step as f64 / num_steps as f64);

        if (step + 1) % 100 == 0 || step < 5 {
            println!(
                "step {:4} / {} | loss {:.4}",
                step + 1,
                num_steps,
                loss.scalar_val()
            );
        }
    }
}

pub fn generate_samples(
    model: &GPT,
    uchars: &[char],
    bos: usize,
    vocab_size: usize,
    block_size: usize,
    num_samples: usize,
    temperature: f64,
) -> Vec<String> {
    println!("\n--- inference ---");
    let mut results = Vec::new();

    for sample_idx in 0..num_samples {
        let mut current_token = bos;
        let mut sample: Vec<char> = Vec::new();
        let mut kv_caches: Option<Vec<(Tensor, Tensor)>> = None;

        for _ in 0..block_size {
            let x = Tensor::new(vec![current_token as f64], vec![1, 1], false);
            let (logits, caches) = model.forward(&x, kv_caches);
            kv_caches = Some(caches);

            let logits_data = logits.data().clone();
            let last = &logits_data[logits_data.len() - vocab_size..];
            let max_l = last.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = last.iter().map(|&v| ((v - max_l) / temperature).exp()).collect();
            let sum_e: f64 = exps.iter().sum();
            let probs: Vec<f64> = exps.iter().map(|&e| e / sum_e).collect();

            let r: f64 = rand_f64();
            let mut cumsum = 0.0f64;
            let mut next_token = 0usize;
            for (v, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    next_token = v;
                    break;
                }
            }

            if next_token == bos {
                break;
            }
            if let Some(&ch) = uchars.get(next_token) {
                sample.push(ch);
            }
            current_token = next_token;
        }

        let name: String = sample.iter().collect();
        results.push(name.clone());
        println!("sample {:2}: {}", sample_idx + 1, name);
    }
    results
}

// tiny thread-local random for generation
fn rand_f64() -> f64 {
    use std::cell::Cell;
    thread_local! { static S: Cell<u64> = Cell::new(98765); }
    S.with(|s| {
        let mut x = s.get();
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x >> 11) as f64 / (1u64 << 53) as f64
    })
}
