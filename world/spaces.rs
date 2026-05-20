//! world/spaces.rs
//! Space definitions for RL environments.

pub mod discrete;
pub mod r#box;

pub use discrete::Discrete;
pub use r#box::Box;