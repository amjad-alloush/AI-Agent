from typing import List, Dict, Any, Optional
from datetime import datetime
import json
class ShortTermMemory:
    """
    Manages short-term memory for the agent, including conversation history
    and recent context.
    """
def __init__(self, max_items: int = 50):
    self.max_items = max_items
    self.conversation_history = []
    self.working_memory = {}
    
def add_interaction(self, user_input: Dict[str, Any], agent_response: Dict[str, Any]) -> None:
    """
        Add a user-agent interaction to the conversation history.
        Args:
        user_input: Processed user input
        agent_response: Agent's response
    """
    timestamp = datetime.now().isoformat()
    interaction = {
    'timestamp': timestamp,
    'user_input': user_input,
    'agent_response': agent_response
    }
    self.conversation_history.append(interaction)
    # Trim history if it exceeds max_items
    if len(self.conversation_history) > self.max_items:
        self.conversation_history = self.conversation_history[-self.max_items:]
        
def get_recent_history(self, n: int = 5) -> List[Dict[str, Any]]:
    """Get the n most recent interactions."""
    return self.conversation_history[-n:] if n < len(self.conversation_history) else self.conversation_history

def set_working_memory(self, key: str, value: Any) -> None:
    """Store a value in working memory."""
    self.working_memory[key] = value

def get_working_memory(self, key: str) -> Optional[Any]:
    """Retrieve a value from working memory."""
    return self.working_memory.get(key)

def clear_working_memory(self) -> None:
    """Clear all working memory."""
    self.working_memory = {}

def get_context_for_reasoning(self) -> Dict[str, Any]:
    """
        Prepare context for the reasoning engine.
        Returns:
        Dictionary containing relevant context from short-term memory
    """
    return {
            'recent_history': self.get_recent_history(),
            'working_memory': self.working_memory
            }

def save_to_file(self, filepath: str) -> None:
    """Save memory state to a file."""
    with open(filepath, 'w') as f:
        json.dump({
        'conversation_history': self.conversation_history,
        'working_memory': self.working_memory
        }, f, indent=2)
    
def load_from_file(self, filepath: str) -> None:
    """Load memory state from a file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
        self.conversation_history = data.get('conversation_history', [])
        self.working_memory = data.get('working_memory', {})