import os
from typing import Dict, Any, List, Optional, Union
import openai


class LLMInterface:
    """
    Interface for interacting with Large Language Models.
    """


def __init__(self, model_name: str = "gpt-3.5-turbo"):
    self.model_name = model_name
    openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_response(
    self,
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Generate a response from the LLM.
    Args:
    messages: List of message dictionaries in OpenAI format
    temperature: Controls randomness (0-1)
    max_tokens: Maximum tokens in response
    Returns:
    Response from the LLM
    """
    try:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return {
            "content": response.choices[0].message.content,
            "model": response.model,
            "usage": response.usage._asdict() if hasattr(response, "usage") else None,
            "finish_reason": response.choices[0].finish_reason,
        }
    except Exception as e:
        return {"error": str(e), "content": f"Error: {str(e)}"}


def generate_with_system_prompt(
    self,
    system_prompt: str,
    user_message: str,
    chat_history: List[Dict[str, str]] = None,
    temperature: float = 0.7,
    max_tokens: int = 1000,
) -> Dict[str, Any]:
    """
    Generate a response with a system prompt and optional chat history.
    Args:
    system_prompt: System instructions for the LLM
    user_message: User's message
    chat_history: Optional previous messages
    temperature: Controls randomness (0-1)
    max_tokens: Maximum tokens in response
    Returns:
    Response from the LLM
    """
    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_message})
    return self.generate_response(messages, temperature, max_tokens)