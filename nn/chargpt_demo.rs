//! nn/chargpt_demo.rs - CharGPT training demo

use ai4::nn::{GPT, Adam, Tensor, Module};
use std::collections::HashMap;
use std::fs;

fn encode(text: &str, char_to_idx: &HashMap<char, usize>) -> Vec<usize> {
    text.chars().filter_map(|c| char_to_idx.get(&c).copied()).collect()
}

fn train_model(
    model: &mut GPT,
    optimizer: &mut Adam,
    data: &[Vec<usize>],
    block_size: usize,
    num_steps: usize,
) {
    for step in 0..num_steps {
        let doc = &data[step % data.len()];
        let n = doc.len().saturating_sub(1).min(block_size);
        if n == 0 { continue; }
        
        let x: Vec<usize> = doc.iter().take(n).cloned().collect();
        let y: Vec<usize> = doc.iter().skip(1).take(n).cloned().collect();
        
        optimizer.zero_grad();
        
        let (logits, _) = model.forward_idx(&x, None);
        let loss = logits.cross_entropy(&y);
        loss.backward();
        
        optimizer.step();
        
        print!("step {:4} / {:4} | loss {:.4}\r", step + 1, num_steps, loss.data.borrow()[0]);
    }
    println!();
}

fn generate_samples(
    model: &GPT,
    idx_to_char: &[char],
    bos: usize,
    vocab_size: usize,
    block_size: usize,
    num_samples: usize,
    temperature: f32,
) {
    println!("\n--- inference ---");
    for _sample_idx in 0..num_samples {
        let mut current_token = bos;
        let mut sample = String::new();
        let mut kv_caches = None;
        
        for _pos_id in 0..block_size {
            let x = vec![current_token];
            let (logits, caches) = model.forward_idx(&x, kv_caches);
            kv_caches = Some(caches);
            
            let data = logits.data.borrow();
            if data.len() < vocab_size { break; }
            
            let logits_slice: Vec<f32> = data[..vocab_size].to_vec();
            
            let max_logit = logits_slice.iter().fold(f32::NEG_INFINITY, |m, &v| m.max(v));
            let exps: Vec<f32> = logits_slice.iter()
                .map(|v| (v - max_logit / temperature).exp())
                .collect();
            let sum: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|v| v / sum).map(|v| if v.is_finite() { v } else { 0.0 }).collect();
            
            let r: f32 = rand::random();
            let mut cumsum = 0.0f32;
            current_token = bos;
            for (i, &p) in probs.iter().enumerate() {
                cumsum += p;
                if r <= cumsum {
                    current_token = i;
                    break;
                }
            }
            
            if current_token == bos {
                break;
            }
            if current_token < idx_to_char.len() {
                sample.push(idx_to_char[current_token]);
            }
        }
        
        println!("sample: {}", sample);
    }
}

fn main() {
    println!("=== CharGPT Demo ===\n");
    
    let data_path = "data/input.txt";
    if !std::path::Path::new(data_path).exists() {
        println!("Downloading names.txt...");
        let _ = std::process::Command::new("curl")
            .args(["-s", "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt", "-o", data_path])
            .output();
    }
    
    let content = fs::read_to_string(data_path).expect("Failed to read data/input.txt");
    let docs: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
    println!("num docs: {}", docs.len());
    
    let mut chars: Vec<char> = docs.iter()
        .flat_map(|s| s.chars())
        .collect();
    chars.sort();
    chars.dedup();
    println!("vocab size: {}", chars.len());
    
    let mut char_to_idx = HashMap::new();
    let mut idx_to_char = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        char_to_idx.insert(c, i);
        idx_to_char.push(c);
    }
    let bos = chars.len();
    let vocab_size = chars.len() + 1;
    
    let block_size = 16;
    let mut model = GPT::new(vocab_size, block_size, 1, 16, 4);
    let mut optimizer = Adam::new(model.parameters(), 0.01, (0.9, 0.999), 1e-8);
    
    let encoded: Vec<Vec<usize>> = docs.iter()
        .map(|doc| {
            let encoded = encode(doc, &char_to_idx);
            let mut full = vec![bos];
            full.extend(encoded);
            full.push(bos);
            full
        })
        .collect();
    
    train_model(&mut model, &mut optimizer, &encoded, block_size, 1000);
    
    generate_samples(&model, &idx_to_char, bos, vocab_size, block_size, 20, 0.5);
}