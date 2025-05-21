"""
Summarizer module for Reasoning Trace Compression.

This module handles the summarization of low-salience segments in reasoning traces
to reduce token usage while preserving logical coherence.
"""

from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TraceSummarizer:
    """
    Summarizes low-salience segments of reasoning traces.
    
    The TraceSummarizer condenses segments that have been identified as less critical
    for the overall logical flow, reducing token usage while preserving meaning.
    """
    
    def __init__(self, use_llm: bool = True, max_summary_ratio: float = 0.3):
        """
        Initialize the TraceSummarizer.
        
        Args:
            use_llm: Whether to use LLM for summarization (if False, uses rule-based)
            max_summary_ratio: Maximum ratio of original length for summaries
        """
        self.use_llm = use_llm
        self.max_summary_ratio = max_summary_ratio
        
        # Initialize LLM if requested
        if self.use_llm:
            try:
                self._initialize_llm()
            except ImportError:
                logger.warning("LLM dependencies not available. Falling back to rule-based summarization.")
                self.use_llm = False
        
        logger.info(f"Initialized TraceSummarizer (use_llm={use_llm})")
    
    def summarize_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Summarize a list of low-salience segments.
        
        Args:
            segments: List of segment dictionaries to summarize
            
        Returns:
            List of summarized segment dictionaries
        """
        summarized_segments = []
        
        for segment in segments:
            # Choose summarization method based on configuration
            if self.use_llm:
                summarized_text = self._summarize_with_llm(segment)
            else:
                summarized_text = self._summarize_with_rules(segment)
            
            # Create new segment with summarized text
            summarized_segment = segment.copy()
            summarized_segment["text"] = summarized_text
            summarized_segment["summarized"] = True
            summarized_segment["original_length"] = len(segment.get("text", ""))
            summarized_segment["summarized_length"] = len(summarized_text)
            
            summarized_segments.append(summarized_segment)
        
        logger.debug(f"Summarized {len(segments)} segments")
        return summarized_segments
    
    def summarize_text(self, text: str) -> str:
        """
        Summarize a single text block.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        if self.use_llm:
            return self._summarize_text_with_llm(text)
        else:
            return self._summarize_text_with_rules(text)
    
    def _summarize_with_rules(self, segment: Dict[str, Any]) -> str:
        """
        Summarize a segment using rule-based techniques.
        
        Args:
            segment: Segment dictionary to summarize
            
        Returns:
            Summarized text
        """
        text = segment.get("text", "")
        segment_type = segment.get("type", "reasoning")
        
        # Different summarization strategies based on segment type
        if segment_type == "evidence":
            return self._summarize_evidence(text)
        elif segment_type == "reasoning":
            return self._summarize_reasoning(text)
        elif segment_type == "step":
            return self._summarize_step(text)
        else:
            # Default summarization for other types
            return self._extract_key_sentences(text)
    
    def _summarize_evidence(self, text: str) -> str:
        """
        Summarize evidence segments by extracting key facts.
        
        Args:
            text: Evidence text to summarize
            
        Returns:
            Summarized evidence
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        # Keep first sentence (usually the main evidence statement)
        if sentences:
            first_sentence = sentences[0]
            
            # If there are multiple sentences, add a brief summary
            if len(sentences) > 1:
                return f"{first_sentence} [Evidence summarized: {len(sentences)} points]"
            else:
                return first_sentence
        
        return text
    
    def _summarize_reasoning(self, text: str) -> str:
        """
        Summarize reasoning segments by focusing on conclusions.
        
        Args:
            text: Reasoning text to summarize
            
        Returns:
            Summarized reasoning
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return text
            
        # For short reasoning, keep as is
        if len(sentences) <= 2:
            return text
            
        # For longer reasoning, keep first and last sentences
        first_sentence = sentences[0]
        last_sentence = sentences[-1]
        
        # If reasoning is very long, add middle summary
        if len(sentences) > 5:
            return f"{first_sentence} [Reasoning steps omitted] {last_sentence}"
        else:
            return f"{first_sentence} {last_sentence}"
    
    def _summarize_step(self, text: str) -> str:
        """
        Summarize step segments by preserving the step number and conclusion.
        
        Args:
            text: Step text to summarize
            
        Returns:
            Summarized step
        """
        # Extract step number if present
        import re
        step_match = re.search(r'(?i)step\s+(\d+):', text)
        step_prefix = f"Step {step_match.group(1)}: " if step_match else ""
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return text
            
        # For short steps, keep as is
        if len(sentences) <= 2:
            return text
            
        # For longer steps, keep conclusion
        last_sentence = sentences[-1]
        
        return f"{step_prefix}{last_sentence}"
    
    def _extract_key_sentences(self, text: str) -> str:
        """
        Extract key sentences from text based on importance markers.
        
        Args:
            text: Text to extract from
            
        Returns:
            Extracted key sentences
        """
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return text
            
        # For short text, keep as is
        if len(sentences) <= 3:
            return text
        
        # Important sentence markers
        importance_markers = [
            "important", "key", "critical", "essential", "significant",
            "therefore", "thus", "hence", "consequently", "as a result",
            "conclude", "conclusion", "in summary", "to summarize"
        ]
        
        # Score sentences by importance
        scored_sentences = []
        for i, sentence in enumerate(sentences):
            score = 0
            
            # Position-based scoring (first and last sentences are important)
            if i == 0 or i == len(sentences) - 1:
                score += 3
                
            # Content-based scoring
            lower_sentence = sentence.lower()
            for marker in importance_markers:
                if marker in lower_sentence:
                    score += 2
                    break
            
            # Length-based penalty (very short sentences might be less informative)
            if len(sentence.split()) < 5:
                score -= 1
                
            scored_sentences.append((sentence, score))
        
        # Sort by score and take top sentences (up to 30% of original)
        max_sentences = max(1, int(len(sentences) * self.max_summary_ratio))
        top_sentences = sorted(scored_sentences, key=lambda x: x[1], reverse=True)[:max_sentences]
        
        # Sort back by original order
        selected_sentences = [s[0] for s in sorted(
            [(s, i) for i, (s, _) in enumerate(top_sentences)], 
            key=lambda x: sentences.index(x[0])
        )]
        
        # Join selected sentences
        return " ".join(selected_sentences)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        import re
        
        # Basic sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out empty sentences
        return [s.strip() for s in sentences if s.strip()]
    
    def _summarize_with_llm(self, segment: Dict[str, Any]) -> str:
        """
        Summarize a segment using LLM.
        
        Args:
            segment: Segment dictionary to summarize
            
        Returns:
            Summarized text
        """
        text = segment.get("text", "")
        segment_type = segment.get("type", "reasoning")
        
        # Create prompt based on segment type
        prompt = self._create_summarization_prompt(text, segment_type)
        
        try:
            # Call LLM for summarization
            summary = self._call_llm_for_summary(prompt)
            
            # Ensure summary is not too long
            max_length = int(len(text) * self.max_summary_ratio)
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
            
        except Exception as e:
            logger.error(f"Error in LLM summarization: {e}")
            logger.info("Falling back to rule-based summarization")
            return self._summarize_with_rules(segment)
    
    def _summarize_text_with_llm(self, text: str) -> str:
        """
        Summarize text using LLM.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        prompt = f"""
        Summarize the following text while preserving key logical points and conclusions:
        
        {text}
        
        Provide a concise summary:
        """
        
        try:
            # Call LLM for summarization
            summary = self._call_llm_for_summary(prompt)
            
            # Ensure summary is not too long
            max_length = int(len(text) * self.max_summary_ratio)
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
            
        except Exception as e:
            logger.error(f"Error in LLM summarization: {e}")
            logger.info("Falling back to rule-based summarization")
            return self._extract_key_sentences(text)
    
    def _summarize_text_with_rules(self, text: str) -> str:
        """
        Summarize text using rule-based techniques.
        
        Args:
            text: Text to summarize
            
        Returns:
            Summarized text
        """
        return self._extract_key_sentences(text)
    
    def _create_summarization_prompt(self, text: str, segment_type: str) -> str:
        """
        Create a prompt for LLM summarization based on segment type.
        
        Args:
            text: Text to summarize
            segment_type: Type of segment
            
        Returns:
            Prompt for LLM
        """
        base_prompt = f"""
        Summarize the following {segment_type} segment while preserving key logical points:
        
        {text}
        
        Provide a concise summary:
        """
        
        # Customize prompt based on segment type
        if segment_type == "hypothesis":
            return base_prompt.replace("Provide a concise summary:", 
                                      "Provide a concise summary of the hypothesis:")
        elif segment_type == "evidence":
            return base_prompt.replace("Provide a concise summary:", 
                                      "Provide a concise summary of the key evidence:")
        elif segment_type == "conclusion":
            return base_prompt.replace("Provide a concise summary:", 
                                      "Provide a concise summary of the conclusion:")
        else:
            return base_prompt
    
    def _initialize_llm(self):
        """Initialize the LLM for summarization."""
        try:
            # This is a placeholder for actual LLM initialization
            # In a real implementation, you would initialize your LLM client here
            # For example, OpenAI API, Hugging Face, etc.
            pass
            
        except ImportError as e:
            logger.error(f"Failed to import LLM dependencies: {e}")
            raise
    
    def _call_llm_for_summary(self, prompt: str) -> str:
        """
        Call LLM to generate a summary.
        
        Args:
            prompt: Prompt for the LLM
            
        Returns:
            Generated summary
        """
        # This is a placeholder for actual LLM API call
        # In a real implementation, you would call your LLM API here
        
        # For now, we'll use a simple extractive approach as a fallback
        text_to_summarize = prompt.split("\n\n")[1].strip()
        return self._extract_key_sentences(text_to_summarize)
