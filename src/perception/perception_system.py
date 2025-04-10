from typing import Dict, Any, Optional, List
from .text_processor import TextProcessor
class PerceptionSystem:
    """
    Manages the processing of various input types.
    """
def __init__(self):
    self.text_processor = TextProcessor()
    # Initialize other processors as needed
    # self.image_processor = ImageProcessor()
    # self.audio_processor = AudioProcessor()
def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
        Process input data based on its type.
        Args:
        input_data: Dictionary containing input data and metadata
        Returns:
        Processed input data
    """
    input_type = input_data.get('type', 'text')
    if input_type == 'text':
      text = input_data.get('content', '')
    return {
        'processed_data': self.text_processor.process(text),
        'type': 'text',
        'metadata': input_data.get('metadata', {})
        }
    # Add handlers for other input types as needed
    # Default fallback
    return {
        'error': f"Unsupported input type: {input_type}",
        'raw_input': input_data
        }

def batch_process(self, inputs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Process multiple inputs in batch."""
    return [self.process_input(input_data) for input_data in inputs]