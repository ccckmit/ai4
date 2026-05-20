//! nn/tests/test_gpt.rs - GPT model tests.

use crate::nn::{GPT, Module, Tensor};

#[test]
fn test_gpt_creation() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let params = gpt.parameters();
    assert!(params.len() > 0);
}

#[test]
fn test_gpt_forward() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let idx = vec![1, 2, 3, 4];
    let (logits, _caches) = gpt.forward_idx(&idx, None);
    assert_eq!(logits.shape.len(), 2);
    assert_eq!(logits.shape[0], 4);
    assert_eq!(logits.shape[1], 20);
}

#[test]
fn test_gpt_kv_cache() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let x1 = vec![1];
    let (_logits1, caches1) = gpt.forward_idx(&x1, None);
    let x2 = vec![2];
    let (logits2, caches2) = gpt.forward_idx(&x2, Some(caches1));
    assert_eq!(logits2.shape.len(), 2);
    assert_eq!(logits2.shape[0], 1);
    assert_eq!(logits2.shape[1], 20);
    assert_eq!(caches2.len(), 1);
}

#[test]
fn test_gpt_multi_layer() {
    let gpt = GPT::new(20, 16, 2, 16, 4);
    let idx = vec![1, 2, 3];
    let (logits, caches) = gpt.forward_idx(&idx, None);
    assert_eq!(logits.shape.len(), 2);
    assert_eq!(logits.shape[0], 3);
    assert_eq!(logits.shape[1], 20);
    assert_eq!(caches.len(), 2);
}