//! world/utils.rs
//! Utility functions for world: registry, random agent.

use std::collections::HashMap;

pub fn registry() -> Vec<String> {
    vec![
        "FrozenLake-v0".to_string(),
        "FrozenLake-v1".to_string(),
        "FrozenLake8x8-v1".to_string(),
        "CartPole-v1".to_string(),
    ]
}

pub fn make(_id: &str) -> Option<super::envs::FrozenLakeEnv> {
    None
}

pub fn run_random_agent(_env_name: &str, _episodes: usize, _render: bool, _seed: Option<u64>) {
    println!("Random agent example:");
    println!("Use: run_random_agent(\"FrozenLake-v1\", 3, false, Some(42))");
}