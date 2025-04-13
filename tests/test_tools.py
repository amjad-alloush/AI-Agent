import pytest
from src.tools.tool_manager import ToolManager
from src.tools.tool_interface import Tool

class MockTool(Tool):
    """Mock tool for testing."""

def __init__(self):
    super().__init__(name="mock_tool", description="A mock tool for testing")


def execute(self, parameters):
    return {
    'success': True,
    'message': f"Executed with parameters: {parameters}"
    }

def test_tool_manager_basic():
    """Test basic functionality of ToolManager."""
    manager = ToolManager()
    tool = MockTool()
    
    # Register tool
    manager.register_tool(tool)
   
   # Get tool
    retrieved_tool = manager.get_tool("mock_tool")
    assert retrieved_tool is not None
    assert retrieved_tool.name == "mock_tool"
    
    # Execute tool
    result = manager.execute_tool("mock_tool", {"param": "value"})
    assert result['success'] is True
    assert "Executed with parameters" in result['message']
    
    # Test non-existent tool
    result = manager.execute_tool("non_existent_tool", {})
    assert 'error' in result
    assert "Tool not found" in result['error']