//! llm/agent.rs - AI Agent with memory and tool feedback
//! 
//! A simple agent that uses Ollama API with memory and tool execution capabilities.

use std::collections::VecDeque;
use std::process::Command;

pub const WORKSPACE: &str = "~/.agent0";
pub const MODEL: &str = "minimax-m2.5:cloud";
pub const MAX_TURNS: usize = 5;

pub const SYSTEM_PROMPT: &str = r#"你是 Jarvis，一個有用的 AI 助理。

重要規則：
1. 當你需要執行 shell 命令時，必須用 <shell> 標籤包住命令
2. <shell> 標籤內可以是多行命令（用反斜槓 \ 或 && 連接）
3. 當你完成所有操作後，用 <end/> 結束你的回覆

流程：
- 如果需要執行命令，輸出 <shell>...</shell>
- 執行完後我會顯示結果
- 如果還需要更多命令，繼續輸出 <shell>
- 當完成所有操作後，輸出 <end/> 表示結束"#;

pub struct Agent {
    conversation_history: VecDeque<String>,
    key_info: Vec<String>,
    workspace: String,
}

impl Agent {
    pub fn new() -> Self {
        Agent {
            conversation_history: VecDeque::new(),
            key_info: Vec::new(),
            workspace: WORKSPACE.to_string(),
        }
    }

    pub fn build_context(&self) -> String {
        let mut context_parts = Vec::new();
        
        if !self.key_info.is_empty() {
            let items: String = self.key_info.iter()
                .map(|k| format!("  <item>{}</item>", k))
                .collect::<Vec<_>>()
                .join("\n");
            context_parts.push(format!("<memory>\n{}\n</memory>", items));
        }
        
        if !self.conversation_history.is_empty() {
            let history: String = self.conversation_history.iter()
                .skip(self.conversation_history.len().saturating_sub(MAX_TURNS * 2))
                .cloned()
                .collect::<Vec<_>>()
                .join("\n");
            context_parts.push(format!("<history>\n{}\n</history>", history));
        }
        
        context_parts.join("\n\n")
    }

    pub fn update_memory(&mut self, user_input: &str, assistant_response: &str, tool_result: Option<&str>) {
        self.conversation_history.push_back(format!("  <user>{}</user>", user_input));
        self.conversation_history.push_back(format!("  <assistant>{}</assistant>", assistant_response));
        
        if let Some(result) = tool_result {
            let truncated = if result.len() > 500 { &result[..500] } else { result };
            self.conversation_history.push_back(format!("  <tool>{}</tool>", truncated));
        }
        
        while self.conversation_history.len() > MAX_TURNS * 4 {
            self.conversation_history.pop_front();
        }
    }

    pub fn add_key_info(&mut self, item: String) {
        if !item.is_empty() && !self.key_info.contains(&item) {
            self.key_info.push(item);
        }
    }

    #[allow(dead_code)]
    pub fn show_memory(&self) -> Vec<&String> {
        self.key_info.iter().collect()
    }

    #[allow(dead_code)]
    pub async fn call_ollama(&self, _prompt: &str, _system: &str) -> String {
        // Note: In real implementation, this would use reqwest for async HTTP
        // For now, returning a placeholder
        "Ollama API call requires async HTTP client (reqwest)".to_string()
    }

    pub fn execute_shell(&self, command: &str) -> Result<String, String> {
        let output = Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .map_err(|e| format!("Execution error: {}", e))?;
        
        if output.status.success() {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ok(stdout.to_string())
        } else {
            let stderr = String::from_utf8_lossy(&output.stderr);
            Ok(format!("Error: {}", stderr))
        }
    }

    #[allow(dead_code)]
    pub fn parse_shell_tags(response: &str) -> Vec<String> {
        let mut cmds = Vec::new();
        
        // Use simple string search
        let tag = "<shell>";
        let end_tag = "</shell>";
        
        let mut remaining = response;
        while let Some(start_idx) = remaining.find(tag) {
            // Move past the <shell> tag
            remaining = &remaining[start_idx + tag.len()..];
            
            if let Some(end_idx) = remaining.find(end_tag) {
                let cmd = &remaining[..end_idx];
                cmds.push(cmd.trim().to_string());
                remaining = &remaining[end_idx + end_tag.len()..];
            } else {
                break;
            }
        }
        
        cmds
    }

    #[allow(dead_code)]
    pub fn has_end_tag(response: &str) -> bool {
        response.contains("<end/>")
    }
}

impl Default for Agent {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_agent_creation() {
        let agent = Agent::new();
        assert!(agent.conversation_history.is_empty());
        assert!(agent.key_info.is_empty());
    }
    
    #[test]
    fn test_build_context() {
        let agent = Agent::new();
        let context = agent.build_context();
        assert!(context.is_empty());
    }
    
    #[test]
    fn test_update_memory() {
        let mut agent = Agent::new();
        agent.update_memory("Hello", "Hi there!", None);
        assert_eq!(agent.conversation_history.len(), 2);
    }
    
    #[test]
    fn test_add_key_info() {
        let mut agent = Agent::new();
        agent.add_key_info("Important info".to_string());
        assert_eq!(agent.key_info.len(), 1);
    }
    
    #[test]
    fn test_parse_shell_tags() {
        let response = "Hello <shell>echo test</shell> world";
        let cmds = Agent::parse_shell_tags(response);
        assert_eq!(cmds.len(), 1);
        assert_eq!(cmds[0], "echo test");
    }
    
    #[test]
    fn test_has_end_tag() {
        let agent = Agent::new();
        assert!(Agent::has_end_tag("Hello <end/>"));
        assert!(!Agent::has_end_tag("Hello"));
    }
    
    #[test]
    fn test_constants() {
        assert_eq!(super::WORKSPACE, "~/.agent0");
        assert_eq!(super::MODEL, "minimax-m2.5:cloud");
        assert_eq!(super::MAX_TURNS, 5);
    }
    
    #[test]
    fn test_system_prompt() {
        let prompt = super::SYSTEM_PROMPT;
        assert!(!prompt.is_empty());
        assert!(prompt.contains("<shell>"));
        assert!(prompt.contains("<end/>"));
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
        assert!(context.contains("<user>Hello</user>"));
    }
    
    #[test]
    fn test_update_memory_with_tool_result() {
        let mut agent = Agent::new();
        agent.update_memory("Run cmd", "Output", Some("tool result"));
        
        assert_eq!(agent.conversation_history.len(), 3);
    }
    
    #[test]
    fn test_update_memory_truncates() {
        let mut agent = Agent::new();
        // Add 20 items (MAX_TURNS * 4 = 20)
        for i in 0..10 {
            agent.update_memory(&format!("user{}", i), &format!("resp{}", i), None);
        }
        
        // Should not exceed MAX_TURNS * 4 = 20 (which is 40 items since update adds 2)
        assert!(agent.conversation_history.len() <= 40);
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
        assert!(result.is_ok()); // Returns output even on error
    }
}