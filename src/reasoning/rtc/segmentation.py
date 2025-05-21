"""
Segmentation module for Reasoning Trace Compression.

This module handles the segmentation of reasoning traces into logical blocks
for subsequent salience scoring and summarization.
"""

from typing import Dict, List, Any, Optional
import re
import logging

logger = logging.getLogger(__name__)

class ReasoningTraceSegmenter:
    """
    Segments reasoning traces into logical blocks based on content and structure.
    
    Segmentation is the first step in the RTC process, breaking down a reasoning trace
    into meaningful units that can be independently evaluated for importance.
    """
    
    def __init__(self):
        """Initialize the ReasoningTraceSegmenter."""
        # Patterns for identifying segment types
        self.segment_patterns = {
            "hypothesis": [
                r"(?i)hypothesis:",
                r"(?i)I think",
                r"(?i)My initial thought",
                r"(?i)Let's assume",
                r"(?i)Let me start by"
            ],
            "evidence": [
                r"(?i)evidence:",
                r"(?i)we know that",
                r"(?i)given that",
                r"(?i)based on",
                r"(?i)according to"
            ],
            "reasoning": [
                r"(?i)reasoning:",
                r"(?i)therefore",
                r"(?i)this implies",
                r"(?i)it follows that",
                r"(?i)we can deduce"
            ],
            "conclusion": [
                r"(?i)conclusion:",
                r"(?i)in conclusion",
                r"(?i)to summarize",
                r"(?i)finally,",
                r"(?i)the answer is"
            ],
            "step": [
                r"(?i)step \d+:",
                r"(?i)first,",
                r"(?i)second,",
                r"(?i)third,",
                r"(?i)next,"
            ]
        }
        
        # Compiled regex patterns for better performance
        self.compiled_patterns = {
            segment_type: [re.compile(pattern) for pattern in patterns]
            for segment_type, patterns in self.segment_patterns.items()
        }
        
        logger.info("Initialized ReasoningTraceSegmenter")
    
    def segment(self, reasoning_trace: str) -> List[Dict[str, Any]]:
        """
        Segment a reasoning trace into logical blocks.
        
        Args:
            reasoning_trace: The reasoning trace to segment
            
        Returns:
            List of segment dictionaries with type, text, and position
        """
        # First, try to segment by explicit markers
        segments = self._segment_by_markers(reasoning_trace)
        
        # If few segments found, try paragraph-based segmentation
        if len(segments) <= 2:
            segments = self._segment_by_paragraphs(reasoning_trace)
        
        # Classify each segment by type
        for segment in segments:
            if "type" not in segment:
                segment["type"] = self._classify_segment(segment["text"])
        
        logger.debug(f"Segmented reasoning trace into {len(segments)} segments")
        return segments
    
    def _segment_by_markers(self, reasoning_trace: str) -> List[Dict[str, Any]]:
        """
        Segment reasoning trace by explicit markers like "Step 1:", "Hypothesis:", etc.
        
        Args:
            reasoning_trace: The reasoning trace to segment
            
        Returns:
            List of segment dictionaries
        """
        segments = []
        
        # Combine all patterns for initial splitting
        all_patterns = []
        for patterns in self.segment_patterns.values():
            all_patterns.extend(patterns)
        
        # Create a regex pattern that matches any of the markers
        combined_pattern = "|".join(f"({pattern})" for pattern in all_patterns)
        compiled_pattern = re.compile(combined_pattern)
        
        # Find all matches
        matches = list(compiled_pattern.finditer(reasoning_trace))
        
        if not matches:
            # No markers found, return the whole trace as one segment
            return [{
                "position": 0,
                "text": reasoning_trace,
                "type": "reasoning"
            }]
        
        # Process matches to create segments
        for i, match in enumerate(matches):
            start_pos = match.start()
            
            # Determine end position (start of next match or end of text)
            end_pos = matches[i+1].start() if i < len(matches) - 1 else len(reasoning_trace)
            
            # Extract segment text including the marker
            segment_text = reasoning_trace[start_pos:end_pos].strip()
            
            # Determine segment type
            segment_type = self._classify_segment(segment_text)
            
            segments.append({
                "position": i,
                "text": segment_text,
                "type": segment_type
            })
        
        # Check if there's content before the first marker
        if matches and matches[0].start() > 0:
            prefix_text = reasoning_trace[:matches[0].start()].strip()
            if prefix_text:
                segments.insert(0, {
                    "position": 0,
                    "text": prefix_text,
                    "type": self._classify_segment(prefix_text)
                })
                
                # Adjust positions of other segments
                for segment in segments[1:]:
                    segment["position"] += 1
        
        return segments
    
    def _segment_by_paragraphs(self, reasoning_trace: str) -> List[Dict[str, Any]]:
        """
        Segment reasoning trace by paragraphs when explicit markers are not found.
        
        Args:
            reasoning_trace: The reasoning trace to segment
            
        Returns:
            List of segment dictionaries
        """
        # Split by double newlines (paragraphs)
        paragraphs = re.split(r'\n\s*\n', reasoning_trace)
        
        segments = []
        for i, paragraph in enumerate(paragraphs):
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            segments.append({
                "position": i,
                "text": paragraph,
                "type": self._classify_segment(paragraph)
            })
        
        return segments
    
    def _classify_segment(self, segment_text: str) -> str:
        """
        Classify a segment based on its content.
        
        Args:
            segment_text: The text of the segment
            
        Returns:
            Segment type as string
        """
        # Check each type's patterns
        for segment_type, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(segment_text):
                    return segment_type
        
        # Default classification based on position and content
        if segment_text.lower().startswith(("i think", "let's", "let me")):
            return "hypothesis"
        elif segment_text.lower().startswith(("therefore", "thus", "so", "hence")):
            return "conclusion"
        elif any(word in segment_text.lower() for word in ["because", "since", "given"]):
            return "evidence"
        else:
            return "reasoning"
