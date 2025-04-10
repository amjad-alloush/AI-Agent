from typing import Dict, Any, List, Optional, Tuple
from .llm_interface import LLMInterface

class ReasoningEngine:
    """
        Core reasoning engine for the AI agent.
    """
def __init__(self, system_prompt: str = None):
    self.llm = LLMInterface()
    # Default system prompt if none provided
    if system_prompt is None:
        self.system_prompt = """
    You are an intelligent AI assistant. Your goal is to understand user requests,
    provide helpful information, and assist with various tasks. You should be polite,
    informative, and helpful while ensuring your responses are accurate and ethical.
    """
    else:
        self.system_prompt = system_prompt


def reason(
self, user_input: Dict[str, Any], memory_context: Dict[str, Any], available_tools: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
        Process user input and generate reasoning.
        Args:
        user_input: Processed user input
        memory_context: Context from memory systems
        available_tools: List of available tools
        Returns:
        Reasoning results including planned actions
    """
    # Extract the user's message
    if 'processed_data' in user_input and 'cleaned_text' in user_input['processed_data']:
        user_message = user_input['processed_data']['cleaned_text']
    else:
        user_message = str(user_input)
    # Prepare context for the LLM
    context = self._prepare_context(user_message, memory_context, available_tools)
    # Generate response from LLM
    llm_response = self.llm.generate_with_system_prompt(
        system_prompt=self.system_prompt + context['system_context'],
        user_message=context['user_message'],
        chat_history=context['chat_history']
    )
    
    # Parse the response to extract reasoning and actions
    parsed_response = self._parse_llm_response(llm_response)
    return {
        'reasoning': parsed_response['reasoning'],
        'planned_actions': parsed_response['actions'],
        'response_to_user': parsed_response['response'],
        'raw_llm_response': llm_response
    }


def _prepare_context( self, user_message: str, memory_context: Dict[str, Any], available_tools: List[Dict[str, Any]] = None ) -> Dict[str, Any]:
    """
        Prepare context for the LLM reasoning.
        Args:
        user_message: User's message
        memory_context: Context from memory systems
        available_tools: List of available tools
        Returns:
        Prepared context for LLM
    """
    # Extract recent conversation history
    chat_history = []
    if 'short_term' in memory_context and 'recent_history' in memory_context['short_term']:
        for interaction in memory_context['short_term']['recent_history']:
            if 'user_input' in interaction and 'processed_data' in  interaction['user_input']:
                user_text = interaction['user_input']['processed_data'].get('cleaned_text', '')
                chat_history.append({"role": "user", "content": user_text})
            if 'agent_response' in interaction and 'response_to_user' in interaction['agent_response']:
                assistant_text = interaction['agent_response']['response_to_user']
                chat_history.append({"role": "assistant", "content": assistant_text})
    
    # Prepare system context with relevant memories
    system_context = "\n\nRelevant information from memory:"
    if 'relevant_memories' in memory_context:
        for i, memory in enumerate(memory_context['relevant_memories']):
            system_context += f"\n- Memory {i+1}: {memory['text']}"
    
    # Add available tools information
    if available_tools:
        system_context += "\n\nAvailable tools:"
        for tool in available_tools:
            system_context += f"\n- {tool['name']}: {tool['description']}"
            
        system_context += "\n\nWhen you need to use a tool, specify it in your response using the format: [USE_TOOL: tool_name, {\"param1\": \"value1\", \"param2\": \"value2\"}]"
    return {
        'system_context': system_context,
        'user_message': user_message,
        'chat_history': chat_history
    }

def _parse_llm_response(self, llm_response: Dict[str, Any]) -> Dict[str, Any]:
    """
        Parse the LLM response to extract reasoning and actions.
        Args:
        llm_response: Raw response from the LLM
        Returns:
        Parsed response with reasoning, actions, and user response
    """
    import re
    content = llm_response.get('content', '')
   
    # Extract tool usage instructions
    tool_pattern = r'\[USE_TOOL: (\w+), ({.*?})\]'
    tool_matches = re.findall(tool_pattern, content)
    actions = []
    for tool_name, params_str in tool_matches:
        try:
            import json
            params = json.loads(params_str)
            actions.append({
            'tool': tool_name,
            'parameters': params
            })
        except json.JSONDecodeError:
            # Handle malformed JSON
            actions.append({
                'tool': tool_name,
                'parameters': params_str,
                'error': 'Invalid JSON parameters'
            })
            
    # Remove tool instructions from the response to user
    response_to_user = re.sub(tool_pattern, '', content).strip()
    
    # Extract reasoning (if present)
    reasoning = content
    reasoning_pattern = r'\[REASONING\](.*?)\[/REASONING\]'
    reasoning_match = re.search(reasoning_pattern, content, re.DOTALL)
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()
    
    # Remove reasoning from response to user
    response_to_user = re.sub(reasoning_pattern, '', response_to_user,
    flags=re.DOTALL).strip()
    return {
        'reasoning': reasoning,
        'actions': actions,
        'response': response_to_user
    }