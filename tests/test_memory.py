import pytest
from src.memory.short_term_memory import ShortTermMemory
from src.memory.long_term_memory import LongTermMemory
import os
import tempfile


def test_short_term_memory_basic():
    """Test basic functionality of ShortTermMemory."""
    memory = ShortTermMemory(max_items=5)

    # Add some interactions
    for i in range(3):
        memory.add_interaction(
        {'processed_data': {'cleaned_text': f"User message {i}"}},
        {'response_to_user': f"Agent response {i}"}
        )
        
    # Check recent history
    history = memory.get_recent_history()
    assert len(history) == 3
    assert history[0]['user_input']['processed_data']['cleaned_text'] == "User message 0"

    # Test max_items limit
    for i in range(3, 8):
        memory.add_interaction(
        {'processed_data': {'cleaned_text': f"User message {i}"}},
        {'response_to_user': f"Agent response {i}"}
        )
    
    history = memory.get_recent_history()
    assert len(history) == 5
    assert history[0]['user_input']['processed_data']['cleaned_text'] == "User message 3"
    
def test_long_term_memory_search():
    """Test search functionality of LongTermMemory."""
    memory = LongTermMemory()
   
   # Add some memories
    memory.add_memory("Python is a programming language.", {"type": "fact"})
    memory.add_memory("The capital of France is Paris.", {"type": "fact"})
    memory.add_memory("Machine learning is a subset of artificial intelligence.",{"type": "fact"})
   
   # Search for memories
    results = memory.search("programming languages")
    assert len(results) > 0
    assert "Python" in results[0]['text']
    results = memory.search("Paris France capital")
    assert len(results) > 0
    assert "France" in results[0]['text']
    assert "Paris" in results[0]['text']

def test_memory_persistence():
    """Test saving and loading memory state."""
    
    # Create temporary directory for test
    with tempfile.TemporaryDirectory() as temp_dir:
        
        # Create and populate memory
        memory = ShortTermMemory()
        memory.add_interaction(
        {'processed_data': {'cleaned_text': "Test message"}},
        {'response_to_user': "Test response"}
        )
        
        # Save state
        save_path = os.path.join(temp_dir, "memory.json")
        memory.save_to_file(save_path)
        
        # Create new memory instance and load state
        new_memory = ShortTermMemory()
        new_memory.load_from_file(save_path)
        
        # Verify loaded state
        history = new_memory.get_recent_history()
        assert len(history) == 1
        assert history[0]['user_input']['processed_data']['cleaned_text'] == "Test message"
        assert history[0]['agent_response']['response_to_user'] == "Test response"