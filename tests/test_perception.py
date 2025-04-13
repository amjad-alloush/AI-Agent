import pytest
from src.perception.text_processor import TextProcessor

def test_text_processor_basic():
    """Test basic functionality of TextProcessor."""
    processor = TextProcessor()
    result = processor.process("Hello, how are you?")
    assert result['cleaned_text'] == "Hello, how are you?"
    assert result['type'] == 'question'
    assert len(result['tokens']) == 5
    
def test_text_processor_command():
    """Test command recognition in TextProcessor."""
    processor = TextProcessor()
    result = processor.process("/weather New York")
    assert result['type'] == 'command'
    assert 'command' in result
    assert result['command']['name'] == 'weather'
    assert result['command']['args'] == 'New York'
    
def test_text_processor_entities():
    """Test entity extraction in TextProcessor."""
    processor = TextProcessor()
    result = processor.process("Check out https://example.com and email me at user@example.com")
    assert len(result['entities']['urls']) == 1
    assert result['entities']['urls'][0] == 'https://example.com'
    assert len(result['entities']['emails']) == 1
    assert result['entities']['emails'][0] == 'user@example.com'