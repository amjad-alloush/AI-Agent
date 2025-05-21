"""
Enhanced Reasoning Engine with Chain-of-Thought capabilities and RTC integration.

This module extends the basic reasoning engine with Chain-of-Thought (CoT) reasoning
and integrates the Reasoning Trace Compression (RTC) mechanism.
"""

from typing import Dict, List, Any, Optional
import logging
import json
import os

from ..tools.tool_manager import ToolManager
from .rtc import ReasoningTraceCompressor

logger = logging.getLogger(__name__)

class ReasoningEngine:
    """
    Enhanced reasoning engine with Chain-of-Thought (CoT) capabilities and RTC integration.
    
    This engine coordinates the reasoning process, including:
    1. Generating Chain-of-Thought reasoning
    2. Compressing reasoning traces when needed
    3. Planning actions based on reasoning
    """
    
    def __init__(self, system_prompt: Optional[str] = None):
        """
        Initialize the ReasoningEngine.
        
        Args:
            system_prompt: Optional system prompt to guide reasoning
        """
        self.system_prompt = system_prompt or self._get_default_system_prompt()
        
        # Initialize RTC
        self.trace_compressor = ReasoningTraceCompressor()
        
        # Track token usage
        self.token_usage = {
            "total_tokens": 0,
            "compressed_tokens": 0,
            "compression_savings": 0
        }
        
        logger.info("Initialized enhanced ReasoningEngine with RTC")
    
    def reason(
        self, 
        user_input: Dict[str, Any], 
        memory_context: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate reasoning and planned actions using Chain-of-Thought.
        
        Args:
            user_input: Processed user input
            memory_context: Context from memory systems
            available_tools: List of available tools
            
        Returns:
            Dictionary containing reasoning results and planned actions
        """
        # 1. Prepare the prompt for reasoning
        prompt = self._prepare_reasoning_prompt(user_input, memory_context, available_tools)
        
        # 2. Generate initial reasoning trace
        initial_reasoning = self._generate_reasoning(prompt)
        
        # 3. Apply RTC if needed
        reasoning_trace, compression_stats = self._apply_rtc_if_needed(initial_reasoning)
        
        # 4. Extract planned actions from reasoning
        planned_actions = self._extract_planned_actions(reasoning_trace, available_tools)
        
        # 5. Generate response to user
        response_to_user = self._generate_user_response(reasoning_trace, planned_actions)
        
        # 6. Update token usage statistics
        self._update_token_usage(compression_stats)
        
        # 7. Return results
        return {
            "reasoning": reasoning_trace,
            "planned_actions": planned_actions,
            "response_to_user": response_to_user,
            "compression_applied": compression_stats.get("compressed", False),
            "token_usage": self.token_usage
        }
    
    def _prepare_reasoning_prompt(
        self, 
        user_input: Dict[str, Any], 
        memory_context: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> str:
        """
        Prepare the prompt for Chain-of-Thought reasoning.
        
        Args:
            user_input: Processed user input
            memory_context: Context from memory systems
            available_tools: List of available tools
            
        Returns:
            Formatted prompt for reasoning
        """
        # Extract user message
        user_message = user_input.get("processed_data", {}).get("cleaned_text", "")
        if not user_message:
            user_message = user_input.get("raw_input", "")
        
        # Format memory context
        formatted_context = self._format_memory_context(memory_context)
        
        # Format available tools
        formatted_tools = self._format_available_tools(available_tools)
        
        # Construct the full prompt
        prompt = f"""
{self.system_prompt}

MEMORY CONTEXT:
{formatted_context}

AVAILABLE TOOLS:
{formatted_tools}

USER INPUT:
{user_message}

REASONING (think step-by-step):
"""
        
        return prompt.strip()
    
    def _format_memory_context(self, memory_context: Dict[str, Any]) -> str:
        """
        Format memory context for inclusion in the prompt.
        
        Args:
            memory_context: Context from memory systems
            
        Returns:
            Formatted memory context
        """
        formatted_parts = []
        
        # Format short-term memory
        if "short_term" in memory_context:
            short_term = memory_context["short_term"]
            formatted_parts.append("Recent conversation:")
            
            for i, interaction in enumerate(short_term.get("interactions", [])[-3:]):
                user_input = interaction.get("user_input", {}).get("processed_data", {}).get("cleaned_text", "")
                agent_response = interaction.get("agent_response", {}).get("response_to_user", "")
                
                formatted_parts.append(f"User: {user_input}")
                formatted_parts.append(f"Assistant: {agent_response}")
        
        # Format relevant long-term memories
        if "relevant_memories" in memory_context:
            memories = memory_context["relevant_memories"]
            if memories:
                formatted_parts.append("\nRelevant past information:")
                
                for i, memory in enumerate(memories):
                    text = memory.get("text", "")
                    formatted_parts.append(f"- {text}")
        
        return "\n".join(formatted_parts)
    
    def _format_available_tools(self, available_tools: List[Dict[str, Any]]) -> str:
        """
        Format available tools for inclusion in the prompt.
        
        Args:
            available_tools: List of available tools
            
        Returns:
            Formatted tools description
        """
        if not available_tools:
            return "No tools available."
        
        formatted_parts = []
        
        for tool in available_tools:
            name = tool.get("name", "")
            description = tool.get("description", "")
            parameters = tool.get("parameters", {})
            
            # Format parameters
            param_str = ""
            if parameters:
                param_parts = []
                for param_name, param_info in parameters.items():
                    param_desc = param_info.get("description", "")
                    param_parts.append(f"  - {param_name}: {param_desc}")
                
                param_str = "\nParameters:\n" + "\n".join(param_parts)
            
            formatted_parts.append(f"Tool: {name}\nDescription: {description}{param_str}\n")
        
        return "\n".join(formatted_parts)
    
    def _generate_reasoning(self, prompt: str) -> str:
        """
        Generate Chain-of-Thought reasoning.
        
        Args:
            prompt: Formatted prompt for reasoning
            
        Returns:
            Generated reasoning trace
        """
        # This is a placeholder for actual LLM call
        # In a real implementation, you would call your LLM API here
        
        # For now, we'll return a mock reasoning trace
        reasoning = """
Let me think through this step by step:

1. First, I need to understand what the user is asking for.
2. Based on the input, they want information about X.
3. I should check if we have any relevant information in memory.
4. Looking at the memory context, I see that we previously discussed Y.
5. I can use the web_search tool to find current information about X.
6. Once I have the information, I'll need to format it clearly for the user.

I'll plan to:
1. Use web_search tool to query "latest information about X"
2. Analyze the results
3. Formulate a comprehensive response
4. Include references to our previous conversation about Y

Therefore, I'll execute the web_search tool and then provide a detailed response.
"""
        
        # Estimate token count (simple approximation)
        token_count = len(reasoning) // 4
        self.token_usage["total_tokens"] += token_count
        
        return reasoning.strip()
    
    def _apply_rtc_if_needed(self, reasoning_trace: str) -> tuple:
        """
        Apply Reasoning Trace Compression if needed.
        
        Args:
            reasoning_trace: Original reasoning trace
            
        Returns:
            Tuple of (compressed_trace, compression_stats)
        """
        # Estimate token count
        token_count = len(reasoning_trace) // 4
        
        # Apply compression if needed
        compressed_trace, stats = self.trace_compressor.compress(
            reasoning_trace=reasoning_trace,
            token_count=token_count
        )
        
        return compressed_trace, stats
    
    def _extract_planned_actions(
        self, 
        reasoning_trace: str, 
        available_tools: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Extract planned actions from reasoning trace.
        
        Args:
            reasoning_trace: Reasoning trace
            available_tools: List of available tools
            
        Returns:
            List of planned actions
        """
        # This is a simplified implementation
        # In a real system, you would use more sophisticated parsing
        
        planned_actions = []
        
        # Look for tool usage patterns in the reasoning
        available_tool_names = [tool.get("name", "") for tool in available_tools]
        
        for tool_name in available_tool_names:
            if tool_name in reasoning_trace:
                # Simple pattern matching for tool usage
                import re
                pattern = rf"{tool_name}\s+(?:with|using)?\s*(?:parameters|params)?:?\s*\{?(.*?)\}?"
                matches = re.findall(pattern, reasoning_trace, re.IGNORECASE | re.DOTALL)
                
                for match in matches:
                    # Try to parse parameters
                    try:
                        # Clean up the match
                        params_str = match.strip()
                        
                        # Handle different formats
                        if params_str.startswith("{") and params_str.endswith("}"):
                            # JSON format
                            parameters = json.loads(params_str)
                        else:
                            # Key-value format
                            parameters = {}
                            param_pairs = params_str.split(",")
                            for pair in param_pairs:
                                if ":" in pair:
                                    key, value = pair.split(":", 1)
                                    parameters[key.strip()] = value.strip()
                        
                        planned_actions.append({
                            "tool": tool_name,
                            "parameters": parameters
                        })
                    except Exception as e:
                        logger.warning(f"Failed to parse parameters for {tool_name}: {e}")
        
        return planned_actions
    
    def _generate_user_response(
        self, 
        reasoning_trace: str, 
        planned_actions: List[Dict[str, Any]]
    ) -> str:
        """
        Generate response to user based on reasoning and planned actions.
        
        Args:
            reasoning_trace: Reasoning trace
            planned_actions: List of planned actions
            
        Returns:
            Response to user
        """
        # This is a placeholder for actual response generation
        # In a real implementation, you would generate a response based on the reasoning
        
        # Extract conclusion from reasoning
        import re
        conclusion_patterns = [
            r"(?i)conclusion:\s*(.*?)(?:\n|$)",
            r"(?i)therefore,\s*(.*?)(?:\n|$)",
            r"(?i)thus,\s*(.*?)(?:\n|$)",
            r"(?i)in summary,\s*(.*?)(?:\n|$)"
        ]
        
        conclusion = ""
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, reasoning_trace)
            if matches:
                conclusion = matches[0]
                break
        
        # If no conclusion found, use the last paragraph
        if not conclusion:
            paragraphs = reasoning_trace.split("\n\n")
            if paragraphs:
                conclusion = paragraphs[-1]
        
        # Format response
        if planned_actions:
            tools_used = ", ".join([action.get("tool", "") for action in planned_actions])
            response = f"{conclusion}\n\nI'll use {tools_used} to help answer your question."
        else:
            response = conclusion
        
        return response
    
    def _update_token_usage(self, compression_stats: Dict[str, Any]) -> None:
        """
        Update token usage statistics.
        
        Args:
            compression_stats: Statistics from compression
        """
        if compression_stats.get("compressed", False):
            original_tokens = compression_stats.get("original_tokens", 0)
            compressed_tokens = compression_stats.get("compressed_tokens", 0)
            tokens_saved = original_tokens - compressed_tokens
            
            self.token_usage["compressed_tokens"] += compressed_tokens
            self.token_usage["compression_savings"] += tokens_saved
            
            logger.info(f"RTC saved {tokens_saved} tokens ({compression_stats.get('compression_ratio', 1.0):.2f}x ratio)")
    
    def _get_default_system_prompt(self) -> str:
        """
        Get the default system prompt for reasoning.
        
        Returns:
            Default system prompt
        """
        return """
You are an intelligent assistant with reasoning capabilities.

When responding to user queries:
1. Think step-by-step to break down complex problems
2. Consider multiple approaches when appropriate
3. Use available tools when needed to gather information
4. Provide clear, accurate, and helpful responses
5. Maintain a friendly and professional tone

Your reasoning should be thorough and logical, leading to well-informed conclusions.
"""
