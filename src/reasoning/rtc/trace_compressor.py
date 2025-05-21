"""
Reasoning Trace Compression (RTC) module.

This module implements the core RTC mechanism described in the LLM-Agent+ paper,
which compresses reasoning traces to improve memory efficiency while preserving
logical coherence.
"""

from typing import Dict, List, Any, Optional, Tuple
import re
import logging

from .segmentation import ReasoningTraceSegmenter
from .salience_scorer import SalienceScorer
from .summarizer import TraceSummarizer
from .consistency_validator import ConsistencyValidator

logger = logging.getLogger(__name__)

class ReasoningTraceCompressor:
    """
    Main class for Reasoning Trace Compression (RTC).
    
    This class coordinates the compression process:
    1. Segmentation: Breaking reasoning traces into logical blocks
    2. Salience Scoring: Ranking blocks by importance
    3. Summarization: Compressing low-salience blocks
    4. Validation: Ensuring logical consistency
    """
    
    def __init__(
        self,
        max_token_threshold: int = 6000,
        target_compression_ratio: float = 3.0,
        min_compression_length: int = 1000,
        preserve_key_segments: bool = True
    ):
        """
        Initialize the ReasoningTraceCompressor.
        
        Args:
            max_token_threshold: Token count that triggers compression (default: 6000)
            target_compression_ratio: Target ratio for compression (default: 3.0)
            min_compression_length: Minimum length before compression is applied (default: 1000)
            preserve_key_segments: Whether to always preserve hypothesis and conclusion segments (default: True)
        """
        self.max_token_threshold = max_token_threshold
        self.target_compression_ratio = target_compression_ratio
        self.min_compression_length = min_compression_length
        self.preserve_key_segments = preserve_key_segments
        
        # Initialize components
        self.segmenter = ReasoningTraceSegmenter()
        self.salience_scorer = SalienceScorer()
        self.summarizer = TraceSummarizer()
        self.validator = ConsistencyValidator()
        
        logger.info(f"Initialized ReasoningTraceCompressor with threshold={max_token_threshold}, "
                   f"target_ratio={target_compression_ratio}")
    
    def should_compress(self, reasoning_trace: str, token_count: int) -> bool:
        """
        Determine if compression should be applied based on token count and trace length.
        
        Args:
            reasoning_trace: The reasoning trace text
            token_count: Estimated token count of the trace
            
        Returns:
            Boolean indicating whether compression should be applied
        """
        if token_count >= self.max_token_threshold:
            logger.info(f"Compression triggered: token count {token_count} exceeds threshold {self.max_token_threshold}")
            return True
        
        if len(reasoning_trace) >= self.min_compression_length and token_count > self.max_token_threshold * 0.75:
            logger.info(f"Compression triggered: approaching threshold with {token_count} tokens")
            return True
            
        return False
    
    def compress(self, reasoning_trace: str, token_count: Optional[int] = None) -> Tuple[str, Dict[str, Any]]:
        """
        Compress a reasoning trace while preserving logical coherence.
        
        Args:
            reasoning_trace: The reasoning trace to compress
            token_count: Optional pre-computed token count
            
        Returns:
            Tuple of (compressed_trace, compression_stats)
        """
        # Estimate token count if not provided
        if token_count is None:
            token_count = self._estimate_token_count(reasoning_trace)
        
        # Check if compression is needed
        if not self.should_compress(reasoning_trace, token_count):
            return reasoning_trace, {
                "compressed": False,
                "original_tokens": token_count,
                "compressed_tokens": token_count,
                "compression_ratio": 1.0
            }
        
        # 1. Segment the reasoning trace
        segments = self.segmenter.segment(reasoning_trace)
        logger.debug(f"Segmented reasoning trace into {len(segments)} segments")
        
        # 2. Score segments by salience
        scored_segments = self.salience_scorer.score_segments(segments)
        
        # 3. Determine which segments to preserve and which to summarize
        segments_to_preserve = []
        segments_to_summarize = []
        
        for segment in scored_segments:
            # Always preserve hypothesis and conclusion if configured
            if self.preserve_key_segments and segment["type"] in ["hypothesis", "conclusion"]:
                segments_to_preserve.append(segment)
            # Preserve high-salience segments
            elif segment["salience_score"] > 0.7:
                segments_to_preserve.append(segment)
            # Summarize low-salience segments
            else:
                segments_to_summarize.append(segment)
        
        # 4. Summarize low-salience segments
        summarized_segments = self.summarizer.summarize_segments(segments_to_summarize)
        
        # 5. Reconstruct the compressed trace
        all_segments = segments_to_preserve + summarized_segments
        all_segments.sort(key=lambda x: x["position"])
        
        compressed_trace = self._reconstruct_trace(all_segments)
        
        # 6. Validate logical consistency
        is_consistent, consistency_score = self.validator.validate(
            original_trace=reasoning_trace,
            compressed_trace=compressed_trace
        )
        
        if not is_consistent:
            logger.warning(f"Compressed trace failed consistency check with score {consistency_score}")
            # Fall back to less aggressive compression if consistency check fails
            return self._fallback_compression(reasoning_trace, token_count)
        
        # 7. Calculate compression statistics
        compressed_token_count = self._estimate_token_count(compressed_trace)
        compression_ratio = token_count / max(1, compressed_token_count)
        
        stats = {
            "compressed": True,
            "original_tokens": token_count,
            "compressed_tokens": compressed_token_count,
            "compression_ratio": compression_ratio,
            "consistency_score": consistency_score,
            "segments_total": len(segments),
            "segments_preserved": len(segments_to_preserve),
            "segments_summarized": len(segments_to_summarize)
        }
        
        logger.info(f"Compressed reasoning trace: {stats['compression_ratio']:.2f}x compression ratio, "
                   f"consistency score: {stats['consistency_score']:.2f}")
        
        return compressed_trace, stats
    
    def _reconstruct_trace(self, segments: List[Dict[str, Any]]) -> str:
        """
        Reconstruct a coherent trace from processed segments.
        
        Args:
            segments: List of segment dictionaries with text content
            
        Returns:
            Reconstructed trace as a string
        """
        # Sort segments by their original position
        sorted_segments = sorted(segments, key=lambda x: x["position"])
        
        # Join segment texts with appropriate transitions
        result_parts = []
        for i, segment in enumerate(sorted_segments):
            # Add transition text for summarized segments
            if segment.get("summarized", False) and i > 0:
                result_parts.append("In summary: ")
            
            result_parts.append(segment["text"])
            
            # Add separator between segments
            if i < len(sorted_segments) - 1:
                result_parts.append("\n\n")
        
        return "".join(result_parts)
    
    def _fallback_compression(self, reasoning_trace: str, token_count: int) -> Tuple[str, Dict[str, Any]]:
        """
        Fallback to a simpler compression strategy when consistency validation fails.
        
        Args:
            reasoning_trace: Original reasoning trace
            token_count: Original token count
            
        Returns:
            Tuple of (compressed_trace, compression_stats)
        """
        logger.info("Using fallback compression strategy")
        
        # Simple fallback: preserve first and last 25% of the trace, summarize the middle
        lines = reasoning_trace.split("\n")
        n_lines = len(lines)
        
        first_quarter = lines[:n_lines//4]
        last_quarter = lines[3*n_lines//4:]
        middle = lines[n_lines//4:3*n_lines//4]
        
        # Summarize the middle section
        middle_text = "\n".join(middle)
        summarized_middle = self.summarizer.summarize_text(middle_text)
        
        # Reconstruct
        compressed_trace = "\n".join(first_quarter) + "\n\n" + summarized_middle + "\n\n" + "\n".join(last_quarter)
        
        # Calculate stats
        compressed_token_count = self._estimate_token_count(compressed_trace)
        compression_ratio = token_count / max(1, compressed_token_count)
        
        stats = {
            "compressed": True,
            "original_tokens": token_count,
            "compressed_tokens": compressed_token_count,
            "compression_ratio": compression_ratio,
            "fallback_used": True
        }
        
        return compressed_trace, stats
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate the number of tokens in a text.
        
        This is a simple approximation. For production, consider using the tokenizer
        from the actual LLM being used.
        
        Args:
            text: Input text
            
        Returns:
            Estimated token count
        """
        # Simple approximation: 4 characters per token on average
        return len(text) // 4
