from typing import Dict, Any, List, Optional
from ..tools.tool_manager import ToolManager


class ActionGenerator:
    """
    Generates and executes actions based on reasoning results.
    """


def __init__(self, tool_manager: ToolManager):
    self.tool_manager = tool_manager


def generate_actions(self, reasoning_results: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate and execute actions based on reasoning results.
    Args:
    reasoning_results: Results from the reasoning engine
    Returns:
    Results of action execution
    """
    planned_actions = reasoning_results.get("planned_actions", [])
    response_to_user = reasoning_results.get("response_to_user", "")

    action_results = []
    # Execute each planned action
    for action in planned_actions:
        tool_name = action.get("tool")
        parameters = action.get("parameters", {})

        if tool_name:
            result = self.tool_manager.execute_tool(tool_name, parameters)
            action_results.append(
                {"tool": tool_name, "parameters": parameters, "result": result}
            )

    return {"response_to_user": response_to_user, "action_results": action_results}


def format_response(self, action_results: Dict[str, Any]) -> str:
    """
    Format the final response to the user, incorporating action results.
    Args:
    action_results: Results from action execution
    Returns:
    Formatted response string
    """
    response = action_results.get("response_to_user", "")
    # Enhance response with action results if needed
    for action in action_results.get("action_results", []):
        tool_name = action.get("tool")
        result = action.get("result", {})

    # Add tool results to response if appropriate
    if tool_name == "weather" and "success" in result and result["success"]:
        weather_data = result.get("data", {})
        location = weather_data.get("location", "")
        temperature = weather_data.get("temperature", "")
        condition = weather_data.get("condition", "")
        weather_info = f"\n\nWeather in {location}: {temperature}°C, {condition}"
        response += weather_info
    elif tool_name == "web_search" and "success" in result and result["success"]:
        # Only add search results if they're not already mentioned in the response
        search_results = result.get("results", [])

        if search_results and "I found some information" not in response:
            response += "\n\nI found some information that might help:"
            for i, search_result in enumerate(search_results[:3]):
                title = search_result.get("title", "")
                link = search_result.get("link", "")
                response += f"\n- {title} ({link})"

    return response