//! tests/test_gpt.rs - GPT model tests.

use ai4::{GPT, Module};

#[test]
fn test_gpt_creation() {
    let gpt = GPT::new(20, 16, 1, 16, 4);
    let params = gpt.parameters();
    assert!(params.len() > 0);
}