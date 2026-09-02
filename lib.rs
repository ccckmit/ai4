//! ai4 - A DIY AI Framework for Rust
//! 
//! A Rust implementation of: RL environments, from-scratch neural networks, and ML toolkit.

#![allow(dead_code, unused, non_snake_case, private_interfaces)]

pub mod world;
pub mod ml;
pub mod nn;
pub mod llm;

pub use world::{Env, Discrete, FrozenLakeEnv, CartPoleEnv, PongEnv, StepResult, registry};
pub use ml::{LinearRegression, LogisticRegression, DecisionTree, RandomForest, KMeans, PCA, StandardScaler, train_test_split, accuracy_score, mean_squared_error, r2_score};
pub use nn::{Tensor, Linear, Embedding, RMSNorm, Adam, GPT, cat};
pub use llm::agent::Agent;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");