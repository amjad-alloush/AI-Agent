from typing import Dict, Any, List, Optional
from .short_term_memory import ShortTermMemory
from .long_term_memory import LongTermMemory


class MemoryManager:
    """
    Manages all memory systems for the agent.
    """


def __init__(self):
    self.short_term = ShortTermMemory()
    self.long_term = LongTermMemory()


def add_interaction(
    self, user_input: Dict[str, Any], agent_response: Dict[str, Any]
) -> None:
    """Add an interaction to short-term memory."""
    self.short_term.add_interaction(user_input, agent_response)


def remember(self, text: str, metadata: Dict[str, Any] = None) -> int:
    """Add information to long-term memory."""
    return self.long_term.add_memory(text, metadata)


def search_memories(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """Search long-term memory."""
    return self.long_term.search(query, top_k)


def get_context(self) -> Dict[str, Any]:
    """
    Get combined context from all memory systems for reasoning.
    Returns:
    Dictionary containing context from all memory systems
    """
    context = {"short_term": self.short_term.get_context_for_reasoning()}
    # Add relevant long-term memories if available
    recent_history = self.short_term.get_recent_history(1)
    if recent_history:
        last_input = recent_history[-1]["user_input"]
    if (
        "processed_data" in last_input
        and "cleaned_text" in last_input["processed_data"]
    ):
        query = last_input["processed_data"]["cleaned_text"]
        relevant_memories = self.long_term.search(query, top_k=3)
        context["relevant_memories"] = relevant_memories

    return context


def save_state(self, directory: str) -> None:
    """Save the state of all memory systems."""
    import os

    os.makedirs(directory, exist_ok=True)
    self.short_term.save_to_file(os.path.join(directory, "short_term.json"))
    self.long_term.save_to_file(os.path.join(directory, "long_term.json"))


def load_state(self, directory: str) -> None:
    """Load the state of all memory systems."""
    import os

    short_term_path = os.path.join(directory, "short_term.json")
    long_term_path = os.path.join(directory, "long_term.json")
    if os.path.exists(short_term_path):
        self.short_term.load_from_file(short_term_path)
    if os.path.exists(long_term_path):
        self.long_term.load_from_file(long_term_path)