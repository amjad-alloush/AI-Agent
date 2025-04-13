from typing import Dict, Any, List, Optional
from .tool_interface import Tool


class ToolManager:
    """
    Manages the registration and execution of tools.
    """


def __init__(self):
    self.tools: Dict[str, Tool] = {}


def register_tool(self, tool: Tool) -> None:
    """
    Register a tool with the manager.
    Args:
    tool: Tool instance to register
    """
    self.tools[tool.name] = tool


def get_tool(self, tool_name: str) -> Optional[Tool]:
    """
    Get a tool by name.
    Args:
    tool_name: Name of the tool to retrieve
    Returns:
    Tool instance if found, None otherwise
    """
    return self.tools.get(tool_name)


def execute_tool(self, tool_name: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a tool by name.
    Args:
    tool_name: Name of the tool to execute
    parameters: Parameters to pass to the tool
    Returns:
    Result of the tool execution
    """
    tool = self.get_tool(tool_name)
    if not tool:
        return {
            "error": f"Tool not found: {tool_name}",
            "available_tools": list(self.tools.keys()),
        }

    try:
        result = tool.execute(parameters)
        return result
    except Exception as e:
        return {
            "error": f"Error executing tool {tool_name}: {str(e)}",
            "parameters": parameters,
        }


def get_available_tools(self) -> List[Dict[str, Any]]:
    """
    Get metadata for all available tools.
    Returns:
    List of tool metadata dictionaries
    """
    return [tool.get_metadata() for tool in self.tools.values()]