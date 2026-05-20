"""Tests for llm/agent.py"""

import pytest
import llm.agent as agent
import re


class TestConstants:
    def test_workspace(self):
        assert agent.WORKSPACE.endswith(".agent0")  # Expanded path

    def test_model(self):
        assert agent.MODEL == "minimax-m2.5:cloud"

    def test_max_turns(self):
        assert agent.MAX_TURNS == 5


class TestSystemPrompt:
    def test_system_prompt_exists(self):
        assert len(agent.SYSTEM_PROMPT) > 0
        assert "<shell>" in agent.SYSTEM_PROMPT
        assert "<end/>" in agent.SYSTEM_PROMPT


class TestBuildContext:
    def test_build_context_empty(self):
        """Test build_context with no history"""
        # Save original state
        orig_history = agent.conversation_history.copy()
        orig_key = agent.key_info.copy()
        
        try:
            agent.conversation_history = []
            agent.key_info = []
            result = agent.build_context()
            assert result == ""
        finally:
            # Restore
            agent.conversation_history = orig_history
            agent.key_info = orig_key

    def test_build_context_with_memory(self):
        """Test build_context with key_info"""
        orig_history = agent.conversation_history.copy()
        orig_key = agent.key_info.copy()
        
        try:
            agent.conversation_history = []
            agent.key_info = ["重要資訊1", "重要資訊2"]
            result = agent.build_context()
            assert "<memory>" in result
            assert "重要資訊1" in result
        finally:
            agent.conversation_history = orig_history
            agent.key_info = orig_key

    def test_build_context_with_history(self):
        """Test build_context with conversation history"""
        orig_history = agent.conversation_history.copy()
        orig_key = agent.key_info.copy()
        
        try:
            agent.conversation_history = [
                "<user>Hello</user>",
                "<assistant>Hi</assistant>",
            ]
            agent.key_info = []
            result = agent.build_context()
            assert "<history>" in result
        finally:
            agent.conversation_history = orig_history
            agent.key_info = orig_key


class TestUpdateMemory:
    def test_update_memory_basic(self):
        """Test update_memory adds to history"""
        orig = agent.conversation_history.copy()
        
        try:
            agent.conversation_history = []
            agent.update_memory("Hello", "Hi there!", None)
            assert len(agent.conversation_history) == 2
        finally:
            agent.conversation_history = orig

    def test_update_memory_with_tool_result(self):
        """Test update_memory with tool result"""
        orig = agent.conversation_history.copy()
        
        try:
            agent.conversation_history = []
            agent.update_memory("Run cmd", "Output", "tool result")
            assert len(agent.conversation_history) == 3
            assert "<tool>" in agent.conversation_history[-1]
        finally:
            agent.conversation_history = orig

    def test_update_memory_truncates(self):
        """Test update_memory truncates long history"""
        orig = agent.conversation_history.copy()
        
        try:
            # Add 20 items (MAX_TURNS * 4 = 20)
            agent.conversation_history = []
            for i in range(20):
                agent.conversation_history.append(f"<user>msg{i}</user>")
            
            agent.update_memory("new", "response", None)
            # Should not exceed MAX_TURNS * 4 = 20
            assert len(agent.conversation_history) <= 20
        finally:
            agent.conversation_history = orig


class TestExtractKeyInfo:
    def test_extract_key_info_regex(self):
        """Test the regex pattern for extracting key info"""
        result = """<memory>
  <item>記住用戶喜歡貓</item>
  <item>用戶住在台北</item>
</memory>"""
        matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
        assert len(matches) == 2

    def test_extract_key_info_empty(self):
        """Test empty memory output"""
        result = "<memory></memory>"
        matches = re.findall(r'<item>(.*?)</item>', result, re.DOTALL)
        assert len(matches) == 0


class TestShellTagParsing:
    def test_parse_single_shell(self):
        """Test parsing single shell command"""
        response = "Hello <shell>echo test</shell> world"
        matches = re.findall(r'<shell>(.+?)</shell>', response, re.DOTALL)
        assert len(matches) == 1
        assert matches[0].strip() == "echo test"

    def test_parse_multiple_shell(self):
        """Test parsing multiple shell commands"""
        response = "<shell>ls</shell> then <shell>cat file.txt</shell>"
        matches = re.findall(r'<shell>(.+?)</shell>', response, re.DOTALL)
        assert len(matches) == 2

    def test_parse_no_shell(self):
        """Test with no shell tags"""
        response = "Just text"
        matches = re.findall(r'<shell>(.+?)</shell>', response, re.DOTALL)
        assert len(matches) == 0


class TestEndTag:
    def test_has_end_tag_true(self):
        """Test end tag detection"""
        response = "Some text <end/> more"
        assert "<end/>" in response

    def test_has_end_tag_false(self):
        """Test no end tag"""
        response = "No end tag here"
        assert "<end/>" not in response


if __name__ == "__main__":
    pytest.main([__file__, "-v"])