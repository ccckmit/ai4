//! world/example.rs - Example of using world environments

use world::envs::{FrozenLakeEnv, CartPoleEnv};
use world::utils::run_random_agent;

fn main() {
    // Example 1: FrozenLake
    println!("=== FrozenLake Example ===");
    let mut env = FrozenLakeEnv::new("4x4", None, true, None);
    let (obs, info) = env.reset(Some(42));
    println!("Initial obs: {}, info: {:?}", obs, info);
    
    for i in 0..5 {
        let result = env.step(i % 4);
        println!("Step {}: obs={}, reward={}, done={}", i+1, result.observation, result.reward, result.done);
    }
    
    // Example 2: Run random agent
    println!("\n=== Random Agent ===");
    run_random_agent("FrozenLake-v1", 3, false, Some(42));
    
    // Example 3: CartPole
    println!("\n=== CartPole Example ===");
    let mut env = CartPoleEnv::new(500, None);
    let (obs, _) = env.reset(Some(0));
    println!("Initial obs: {:?}", obs);
    
    for i in 0..3 {
        let result = env.step(i % 2);
        println!("Step {}: reward={}, done={}", i+1, result.reward, result.done);
    }
}