from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
class Tool(ABC):
    """
    Abstract base class for all tools.
    """
    
def __init__(self, name: str, description: str):
    self.name = name
    self.description = description

@abstractmethod
def execute(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """
        Execute the tool with the given parameters.
        Args:
        parameters: Tool-specific parameters
        Returns:
        Result of the tool execution
    """
pass

def get_metadata(self) -> Dict[str, Any]:
    """
        Get tool metadata.
        Returns:
        Dictionary containing tool metadata
    """
    return {
        'name': self.name,
        'description': self.description
    }