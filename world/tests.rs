//! world/tests.rs
//! Basic smoke tests for world.

use world::spaces::{Discrete, Box};
use world::utils::run_random_agent;

#[test]
fn test_discrete_space() {
    let mut sp = Discrete::new(4, 0, Some(0));
    for _ in 0..50 {
        let s = sp.sample();
        assert!(sp.contains(&s), "{} not in Discrete", s);
    }
    assert!(sp.contains(&0));
    assert!(sp.contains(&3));
    assert!(!sp.contains(&4));
    assert!(!sp.contains(&999));
}

#[test]
fn test_box_space() {
    let mut sp = Box::new(vec![-1.0, -1.0, -1.0], vec![1.0, 1.0, 1.0], Some(vec![3]), "f32", Some(0));
    for _ in 0..50 {
        let s = sp.sample();
        assert!(sp.contains(&s), "{:?} not in Box", s);
    }
}

#[test]
fn test_frozen_lake_reset() {
    use world::envs::FrozenLakeEnv;
    let mut env = FrozenLakeEnv::new("4x4", None, true, None);
    let (obs, info) = env.reset(Some(0));
    assert_eq!(obs, 0, "Expected start state 0, got {}", obs);
}

#[test]
fn test_frozen_lake_step() {
    use world::envs::FrozenLakeEnv;
    let mut env = FrozenLakeEnv::new("4x4", None, false, None);
    let (obs, _) = env.reset(Some(1));
    let result = env.step(2);
    assert!(result.observation >= 0);
}

#[test]
fn test_frozen_lake_unpack() {
    use world::envs::FrozenLakeEnv;
    let mut env = FrozenLakeEnv::new("4x4", None, false, None);
    env.reset(Some(0));
    let result = env.step(1);
    assert!(result.observation >= 0);
}

#[test]
fn test_frozen_lake_full_episode() {
    use world::envs::FrozenLakeEnv;
    let mut env = FrozenLakeEnv::new("4x4", None, false, None);
    let (obs, _) = env.reset(Some(42));
    let mut done = false;
    let mut steps = 0;
    while !done && steps < 200 {
        let result = env.step(steps % 4);
        done = result.done();
        steps += 1;
    }
    assert!(steps > 0);
}

#[test]
fn test_cartpole_reset() {
    use world::envs::CartPoleEnv;
    let mut env = CartPoleEnv::new(500, None);
    let (obs, _info) = env.reset(Some(0));
    assert_eq!(obs.len(), 4, "Expected shape (4,), got {:?}", obs);
}

#[test]
fn test_cartpole_step() {
    use world::envs::CartPoleEnv;
    let mut env = CartPoleEnv::new(500, None);
    env.reset(Some(0));
    let result = env.step(0);
    assert_eq!(result.observation.len(), 4);
    assert_eq!(result.reward, 1.0);
}

#[test]
fn test_cartpole_full_episode() {
    use world::envs::CartPoleEnv;
    let mut env = CartPoleEnv::new(500, None);
    let (obs, _) = env.reset(Some(7));
    let mut total_reward = 0.0;
    let mut done = false;
    while !done {
        let result = env.step(0);
        total_reward += result.reward;
        done = result.done();
    }
    assert!(total_reward >= 1.0);
}

#[test]
fn test_cartpole_invalid_action() {
    use world::envs::CartPoleEnv;
    let mut env = CartPoleEnv::new(500, None);
    let (_, _) = env.reset(Some(0));
}

#[test]
fn test_time_limit_wrapper() {
    use world::envs::CartPoleEnv;
    use world::wrappers::TimeLimitWrapper;
    let env = CartPoleEnv::new(500, None);
    let mut wrapped = TimeLimitWrapper::new(Box::new(env), 5);
    wrapped.reset(Some(0));
    
    for _ in 0..4 {
        let result = wrapped.step(0);
        assert!(!result.truncated || result.terminated);
    }
    
    let result = wrapped.step(0);
    assert!(result.truncated || result.terminated);
}

#[test]
fn test_registry() {
    let r = world::registry();
    assert!(r.contains(&"FrozenLake-v0".to_string()));
    assert!(r.contains(&"FrozenLake-v1".to_string()));
    assert!(r.contains(&"CartPole-v1".to_string()));
}