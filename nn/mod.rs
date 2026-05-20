//! nn - DIY Neural Network framework with autograd.

pub mod tensor;
pub mod nn;
pub mod gpt;

pub use tensor::{Tensor, cat};
pub use nn::{Module, Linear, Embedding, RMSNorm, Adam};
pub use gpt::GPT;