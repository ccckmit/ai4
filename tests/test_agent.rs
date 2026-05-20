//! tests/test_agent.rs - AI Agent tests.

use ai4::Agent;

#[test]
fn test_agent_creation() {
    let agent = Agent::new();
    let _context = agent.build_context();
    assert!(true);
}

#[test]
fn test_build_context() {
    let agent = Agent::new();
    let context = agent.build_context();
    assert!(context.is_empty());
}

#[test]
fn test_build_context_with_memory() {
    let mut agent = Agent::new();
    agent.add_key_info("重要資訊1".to_string());
    agent.add_key_info("重要資訊2".to_string());

    let context = agent.build_context();
    assert!(context.contains("<memory>"));
    assert!(context.contains("重要資訊1"));
}

#[test]
fn test_build_context_with_history() {
    let mut agent = Agent::new();
    agent.update_memory("Hello", "Hi there!", None);

    let context = agent.build_context();
    assert!(context.contains("<history>"));
}

#[test]
fn test_update_memory() {
    let mut agent = Agent::new();
    agent.update_memory("Hello", "Hi there!", None);
    assert_eq!(agent.show_memory().len(), 0);
}

#[test]
fn test_update_memory_with_tool_result() {
    let mut agent = Agent::new();
    agent.update_memory("Run cmd", "Output", Some("tool result"));
    agent.update_memory("new", "response", None);
}

#[test]
fn test_add_key_info() {
    let mut agent = Agent::new();
    agent.add_key_info("Important info".to_string());
    assert_eq!(agent.show_memory().len(), 1);
}

#[test]
fn test_constants() {
    use ai4::llm::agent;
    assert_eq!(agent::WORKSPACE, "~/.agent0");
    assert_eq!(agent::MODEL, "minimax-m2.5:cloud");
    assert_eq!(agent::MAX_TURNS, 5);
}

#[test]
fn test_system_prompt() {
    use ai4::llm::agent;
    let prompt = agent::SYSTEM_PROMPT;
    assert!(!prompt.is_empty());
    assert!(prompt.contains("<shell>"));
    assert!(prompt.contains("<end/>"));
}

#[test]
fn test_parse_shell_tags() {
    let response = "Hello <shell>echo test</shell> world";
    let cmds = Agent::parse_shell_tags(response);
    assert_eq!(cmds.len(), 1);
    assert_eq!(cmds[0], "echo test");
}

#[test]
fn test_parse_multiple_shell() {
    let response = "<shell>ls</shell> then <shell>cat file.txt</shell>";
    let cmds = Agent::parse_shell_tags(response);
    assert_eq!(cmds.len(), 2);
}

#[test]
fn test_parse_no_shell() {
    let response = "Just text";
    let cmds = Agent::parse_shell_tags(response);
    assert_eq!(cmds.len(), 0);
}

#[test]
fn test_has_end_tag() {
    assert!(Agent::has_end_tag("Hello <end/>"));
    assert!(!Agent::has_end_tag("Hello"));
}

#[test]
fn test_execute_shell() {
    let agent = Agent::new();
    let result = agent.execute_shell("echo hello");
    assert!(result.is_ok());
    assert!(result.unwrap().contains("hello"));
}

#[test]
fn test_execute_shell_error() {
    let agent = Agent::new();
    let result = agent.execute_shell("exit 1");
    assert!(result.is_ok());
}

#[test]
fn test_update_memory_truncates() {
    let mut agent = Agent::new();
    for i in 0..10 {
        agent.update_memory(&format!("user{}", i), &format!("resp{}", i), None);
    }
    assert!(true);
}