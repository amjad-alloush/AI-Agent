import pytest
from src.agent import Agent

def test_agent_basic_interaction():
    """Test basic interaction with the agent."""
    agent = Agent()
    # Process a simple greeting
    response = agent.process_input({
    'type': 'text',
    'content': 'Hello, how are you?',
    'metadata': {'source': 'test'}
    })
    assert 'response' in response
    assert isinstance(response['response'], str)
    assert len(response['response']) > 0


def test_agent_tool_use():
    """Test the agent's ability to use tools."""
    agent = Agent()
    # Process a query that should trigger tool use
    response = agent.process_input({
    'type': 'text',
    'content': 'What\'s the weather in New York?',
    'metadata': {'source': 'test'}
    })
    assert 'response' in response
    assert 'actions' in response
    # Check if weather tool was used
    tool_used = False
    for action in response['actions']:
        if action.get('tool') == 'weather':
            tool_used = True
            break
        
    assert tool_used, "Weather tool should have been used"

def test_agent_memory():
    """Test the agent's memory capabilities."""
    agent = Agent()
    
    # First interaction
    agent.process_input({
    'type': 'text',
    'content': 'My name is John',
    'metadata': {'source': 'test'}
    })
    
    # Second interaction should reference the first
    response = agent.process_input({
    'type': 'text',
    'content': 'What\'s my name?',
    'metadata': {'source': 'test'}
    })
    assert 'John' in response['response'], "Agent should remember the user's name"