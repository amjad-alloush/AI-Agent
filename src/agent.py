from typing import Dict, Any, List, Optional
import os
import json
from .perception.perception_system import PerceptionSystem
from .memory.memory_manager import MemoryManager
from .reasoning.reasoning_engine import ReasoningEngine
from .tools.tool_manager import ToolManager
from .tools.web_search_tool import WebSearchTool
from .tools.weather_tool import WeatherTool
from .action.action_generator import ActionGenerator


class Agent:
    """
    Main AI agent class that coordinates all components.
    """


def __init__(self, system_prompt: str = None):
    # Initialize components
    self.perception = PerceptionSystem()
    self.memory = MemoryManager()
    self.reasoning = ReasoningEngine(system_prompt)

    # Initialize tool manager and register tools
    self.tool_manager = ToolManager()
    self._register_default_tools()

    # Initialize action generator
    self.action_generator = ActionGenerator(self.tool_manager)


def _register_default_tools(self) -> None:
    """Register default tools with the tool manager."""
    # Register web search tool
    web_search_api_key = os.getenv("SEARCH_API_KEY")
    self.tool_manager.register_tool(WebSearchTool(api_key=web_search_api_key))

    # Register weather tool
    weather_api_key = os.getenv("WEATHER_API_KEY")
    self.tool_manager.register_tool(WeatherTool(api_key=weather_api_key))
    # Register additional tools as needed


def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process user input and generate a response.
    Args:
    input_data: Dictionary containing input data
    Returns:
    Agent's response
    """

    # 1. Perception: Process the input
    processed_input = self.perception.process_input(input_data)

    # 2. Memory: Get context from memory
    memory_context = self.memory.get_context()

    # 3. Reasoning: Generate reasoning and planned actions
    available_tools = self.tool_manager.get_available_tools()
    reasoning_results = self.reasoning.reason(
        processed_input, memory_context, available_tools
    )

    # 4. Action: Execute planned actions
    action_results = self.action_generator.generate_actions(reasoning_results)

    # 5. Format the final response
    response_text = self.action_generator.format_response(action_results)

    # 6. Update memory with this interaction
    agent_response = {
        "response_to_user": response_text,
        "reasoning": reasoning_results.get("reasoning", ""),
        "actions": action_results.get("action_results", []),
    }
    self.memory.add_interaction(processed_input, agent_response)

    # 7. Store important information in long-term memory if needed
    if (
        "processed_data" in processed_input
        and "cleaned_text" in processed_input["processed_data"]
    ):
        user_text = processed_input["processed_data"]["cleaned_text"]

    # Store important interactions in long-term memory
    if len(user_text.split()) > 5:  # Only store substantial messages
        self.memory.remember(
            f"User: {user_text}\nAssistant: {response_text}", {"type": "interaction"}
        )

    return {
        "response": response_text,
        "processed_input": processed_input,
        "reasoning": reasoning_results.get("reasoning", ""),
        "actions": action_results.get("action_results", []),
    }


def save_state(self, directory: str) -> None:
    """
    Save the agent's state to disk.
    Args:
    directory: Directory to save state in
    """
    os.makedirs(directory, exist_ok=True)

    # Save memory state
    self.memory.save_state(os.path.join(directory, "memory"))

    # Save agent configuration
    config = {
        "system_prompt": self.reasoning.system_prompt,
        "tools": [tool.get_metadata() for tool in self.tool_manager.tools.values()],
    }
    with open(os.path.join(directory, "agent_config.json"), "w") as f:
        json.dump(config, f, indent=2)


def load_state(self, directory: str) -> None:
    """
    Load the agent's state from disk.
    Args:
    directory: Directory to load state from
    """
    # Load memory state
    memory_dir = os.path.join(directory, "memory")
    if os.path.exists(memory_dir):
        self.memory.load_state(memory_dir)

    # Load agent configuration
    config_path = os.path.join(directory, "agent_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = json.load(f)

    # Update system prompt if available
    if "system_prompt" in config:
        self.reasoning.system_prompt = config["system_prompt"]
