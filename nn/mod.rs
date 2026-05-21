//! nn - DIY Neural Network framework with autograd.

pub mod tensor;
pub mod nn;
pub mod gpt;
pub mod chargpt;

pub use tensor::{Tensor, cat};
pub use nn::{Linear, Embedding, RMSNorm, Adam};
pub use gpt::GPT;
pub use chargpt::{generate_samples, train_model};pub mod cnn;
pub use cnn::*;

#[cfg(test)]
mod tests {
    mod test_cnn;
    mod test_cnn_gemini;
}
