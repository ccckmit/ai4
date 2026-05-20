//! world/envs.rs
//! Built-in RL environments.

pub mod frozen_lake;
pub mod cartpole;

pub use frozen_lake::FrozenLakeEnv;
pub use cartpole::CartPoleEnv;