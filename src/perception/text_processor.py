import re
from typing import Dict, Any


class TextProcessor:
    """
    Processes text inputs for the AI agent.
    """


def __init__(self):
    self.patterns = {
        "command": re.compile(r"^/(\w+)(?:\s+(.*))?$"),
        "question": re.compile(r".*\?$"),
        "url": re.compile(r"https?://\S+"),
        "email": re.compile(r"\S+@\S+\.\S+"),
    }


def process(self, text: str) -> Dict[str, Any]:
    """
    Process text input and extract structured information.
    Args:
    text: The raw text input from the user
    Returns:
        A dictionary containing processed information
    """
    # Basic preprocessing
    cleaned_text = text.strip()
    # Extract entities and patterns
    entities = self._extract_entities(cleaned_text)
    # Determine input type
    input_type = self._determine_input_type(cleaned_text)
    # Create structured representation
    result = {
        "raw_text": text,
        "cleaned_text": cleaned_text,
        "type": input_type,
        "entities": entities,
        "tokens": cleaned_text.split(),
        "length": len(cleaned_text),
    }

    # Extract command if present
    command_match = self.patterns["command"].match(cleaned_text)
    if command_match:
        result["command"] = {
            "name": command_match.group(1),
            "args": command_match.group(2) if command_match.group(2) else "",
        }
    return result


def _extract_entities(self, text: str) -> Dict[str, list]:
    # Extract entities like URLs, emails, etc. from text.
    entities = {
        "urls": re.findall(self.patterns["url"], text),
        "emails": re.findall(self.patterns["email"], text),
    }
    return entities


def _determine_input_type(self, text: str) -> str:
    """Determine the type of input based on patterns."""
    if self.patterns["command"].match(text):
        return "command"
    elif self.patterns["question"].match(text):
        return "question"
    elif len(text.split()) <= 3:
        return "keyword"
    else:
        return "statement"