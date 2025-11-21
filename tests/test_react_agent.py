"""
Pytest Tests for React Agent (Real API Calls)
==============================================
Tests LangGraph agent with tools using actual API.
"""

import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from react_agent import (
    calculator,
    get_word_length,
    reverse_string,
    create_agent,
    should_continue
)
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def agent():
    """Create agent instance"""
    return create_agent()


# ============================================================================
# TOOL TESTS
# ============================================================================

class TestTools:
    """Test individual tools work correctly"""
    
    def test_calculator_addition(self):
        """Test calculator with addition"""
        result = calculator.invoke({"expression": "5 + 3"})
        assert "8" in result
    
    def test_calculator_multiplication(self):
        """Test calculator with multiplication"""
        result = calculator.invoke({"expression": "7 * 6"})
        assert "42" in result
    
    def test_calculator_complex_expression(self):
        """Test calculator with complex expression"""
        result = calculator.invoke({"expression": "(10 + 5) * 2"})
        assert "30" in result
    
    def test_calculator_error_handling(self):
        """Test calculator handles invalid input"""
        result = calculator.invoke({"expression": "invalid"})
        assert "Error" in result
    
    def test_word_length(self):
        """Test word length counter"""
        result = get_word_length.invoke({"word": "artificial"})
        assert "10" in result
        assert "artificial" in result.lower()
    
    def test_word_length_short_word(self):
        """Test word length with short word"""
        result = get_word_length.invoke({"word": "AI"})
        assert "2" in result
    
    def test_reverse_string(self):
        """Test string reversal"""
        result = reverse_string.invoke({"text": "Hello"})
        assert "olleH" in result
    
    def test_reverse_string_with_spaces(self):
        """Test reverse with spaces"""
        result = reverse_string.invoke({"text": "Hello World"})
        assert "dlroW olleH" in result


# ============================================================================
# AGENT TESTS
# ============================================================================

class TestAgent:
    """Test agent with real API calls"""
    
    def test_agent_creation(self, agent):
        """Test agent is created successfully"""
        assert agent is not None
    
    def test_agent_calculator_query(self, agent):
        """Test agent uses calculator tool"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="What is 25 multiplied by 4?")
            ]
        })
        
        messages = result["messages"]
        assert len(messages) >= 2
        
        # Find the final answer
        final_message = messages[-1]
        final_content = final_message.content.lower()
        
        # Should mention 100
        assert "100" in final_content
    
    def test_agent_word_length_query(self, agent):
        """Test agent uses word length tool"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="How many letters are in the word 'computer'?")
            ]
        })
        
        messages = result["messages"]
        final_message = messages[-1]
        final_content = final_message.content.lower()
        
        # Should mention 8
        assert "8" in final_content or "eight" in final_content
    
    def test_agent_reverse_string_query(self, agent):
        """Test agent uses reverse string tool"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="Can you reverse the text 'Python'?")
            ]
        })
        
        messages = result["messages"]
        final_message = messages[-1]
        final_content = final_message.content
        
        # Should mention nohtyP
        assert "nohtyP" in final_content or "nohtyp" in final_content.lower()
    
    def test_agent_general_knowledge_query(self, agent):
        """Test agent answers without tools"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Answer questions directly when tools aren't needed."),
                HumanMessage(content="What is the capital of France?")
            ]
        })
        
        messages = result["messages"]
        final_message = messages[-1]
        final_content = final_message.content.lower()
        
        # Should answer Paris without using tools
        assert "paris" in final_content
    
    def test_agent_handles_multiple_steps(self, agent):
        """Test agent can handle multi-step reasoning"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="Calculate 15 + 25, then tell me how many letters are in 'testing'")
            ]
        })
        
        messages = result["messages"]
        
        # Should have used tools
        tool_calls_found = False
        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tool_calls_found = True
                break
        
        assert tool_calls_found


# ============================================================================
# ROUTING TESTS
# ============================================================================

class TestRouting:
    """Test should_continue routing function"""
    
    def test_should_continue_with_tool_calls(self):
        """Test routing continues when tools are called"""
        # Create a message with tool calls
        msg = AIMessage(content="", tool_calls=[
            {"name": "calculator", "args": {"expression": "2+2"}, "id": "call_123"}
        ])
        
        state = {"messages": [msg]}
        result = should_continue(state)
        
        assert result == "continue"
    
    def test_should_end_without_tool_calls(self):
        """Test routing ends when no tools are called"""
        msg = AIMessage(content="The answer is 4")
        state = {"messages": [msg]}
        
        result = should_continue(state)
        
        assert result == "end"


# ============================================================================
# PARAMETRIZED TESTS
# ============================================================================

class TestVariousQueries:
    """Test agent with various query types"""
    
    @pytest.mark.parametrize("query,expected_in_response", [
        ("What is 10 times 5?", ["50"]),
        ("How many letters in 'hello'?", ["5", "five"]),
        ("Reverse 'test'", ["tset"]),
    ])
    def test_various_tool_queries(self, agent, query, expected_in_response):
        """Test agent handles various queries correctly"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content=query)
            ]
        })
        
        final_content = result["messages"][-1].content.lower()
        
        # At least one expected term should be in response
        assert any(expected in final_content for expected in expected_in_response)


# ============================================================================
# EDGE CASES
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""
    
    def test_agent_invalid_calculation(self, agent):
        """Test agent handles invalid math"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="What is 5 divided by zero?")
            ]
        })
        
        # Should complete without crashing
        assert result is not None
        assert "messages" in result
    
    def test_agent_empty_word(self, agent):
        """Test agent handles empty string"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="How many letters in ''?")
            ]
        })
        
        # Should handle gracefully
        assert result is not None
    
    def test_agent_ambiguous_query(self, agent):
        """Test agent handles ambiguous query"""
        result = agent.invoke({
            "messages": [
                SystemMessage(content="Use tools when needed."),
                HumanMessage(content="Calculate something")
            ]
        })
        
        # Should respond without crashing
        assert result is not None
        messages = result["messages"]
        assert len(messages) > 0


# ============================================================================
# SKIP IF NO API KEY
# ============================================================================

def pytest_collection_modifyitems(config, items):
    """Skip tests requiring API key if not available"""
    if not os.getenv("OPENAI_API_KEY"):
        skip_api = pytest.mark.skip(reason="OPENAI_API_KEY not set")
        for item in items:
            item.add_marker(skip_api)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

