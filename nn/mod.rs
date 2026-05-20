//! nn - DIY Neural Network framework with autograd.

pub mod tensor;
pub mod nn;
pub mod gpt;
pub mod cnn;
pub mod datasets;

pub use tensor::{Tensor, cat};
pub use nn::{Module, Linear, Embedding, RMSNorm, Adam};
pub use gpt::GPT;
pub use cnn::{Conv2d, MaxPool2d, AvgPool2d, Flatten, BatchNorm2d, Dropout2d};
pub use datasets::{Dataset, DataLoader, load_mnist};

#[cfg(test)]
mod tests {
    mod test_tensor;
    mod test_nn;
    mod test_gpt;
    mod test_by_claude;
    mod test_cnn;
    mod test_cnn_gemini;
}