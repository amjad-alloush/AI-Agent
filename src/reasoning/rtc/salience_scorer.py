"""
Salience Scorer module for Reasoning Trace Compression.

This module handles the scoring of reasoning trace segments based on their importance
for maintaining logical coherence during compression.
"""

from typing import Dict, List, Any, Optional
import logging
import re

logger = logging.getLogger(__name__)

class SalienceScorer:
    """
    Scores reasoning trace segments based on their importance for logical coherence.
    
    The SalienceScorer assigns importance scores to each segment, which determines
    whether they should be preserved verbatim or summarized during compression.
    """
    
    def __init__(self, use_bert: bool = False):
        """
        Initialize the SalienceScorer.
        
        Args:
            use_bert: Whether to use BERT-based scoring (requires additional dependencies)
        """
        self.use_bert = use_bert
        
        # Keywords that indicate high salience
        self.high_salience_keywords = [
            "therefore", "thus", "hence", "conclusion", "answer", 
            "important", "critical", "key", "essential", "crucial",
            "because", "since", "due to", "result", "consequently"
        ]
        
        # Initialize BERT model if requested
        if self.use_bert:
            try:
                self._initialize_bert_model()
            except ImportError:
                logger.warning("BERT dependencies not available. Falling back to rule-based scoring.")
                self.use_bert = False
        
        logger.info(f"Initialized SalienceScorer (use_bert={use_bert})")
    
    def score_segments(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score segments based on their importance for logical coherence.
        
        Args:
            segments: List of segment dictionaries from the segmenter
            
        Returns:
            List of segments with added salience_score field (0.0-1.0)
        """
        # Choose scoring method based on configuration
        if self.use_bert:
            return self._score_with_bert(segments)
        else:
            return self._score_with_rules(segments)
    
    def _score_with_rules(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score segments using rule-based heuristics.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segments with added salience_score field
        """
        scored_segments = []
        
        # First pass: assign base scores by segment type
        for segment in segments:
            segment_type = segment.get("type", "reasoning")
            
            # Base score by segment type
            if segment_type == "hypothesis":
                base_score = 0.9  # Hypotheses are usually important
            elif segment_type == "conclusion":
                base_score = 0.95  # Conclusions are almost always important
            elif segment_type == "evidence":
                base_score = 0.8  # Evidence is usually important
            elif segment_type == "step" and len(segments) <= 5:
                base_score = 0.8  # In short traces, steps are important
            else:
                base_score = 0.5  # Default score for other segments
            
            # Adjust score based on position
            position_factor = self._calculate_position_factor(segment, segments)
            
            # Adjust score based on content
            content_factor = self._calculate_content_factor(segment)
            
            # Calculate final score
            final_score = min(1.0, base_score * position_factor * content_factor)
            
            # Add score to segment
            segment["salience_score"] = final_score
            scored_segments.append(segment)
        
        # Second pass: ensure logical flow by adjusting adjacent segments
        self._adjust_for_logical_flow(scored_segments)
        
        return scored_segments
    
    def _calculate_position_factor(self, segment: Dict[str, Any], all_segments: List[Dict[str, Any]]) -> float:
        """
        Calculate position-based adjustment factor.
        
        Args:
            segment: The segment to score
            all_segments: All segments in the trace
            
        Returns:
            Position adjustment factor (0.8-1.2)
        """
        position = segment.get("position", 0)
        total_segments = len(all_segments)
        
        # First and last segments are usually more important
        if position == 0 or position == total_segments - 1:
            return 1.2
        
        # Middle segments slightly less important
        middle_position = total_segments / 2
        distance_from_middle = abs(position - middle_position) / middle_position
        
        # Closer to the middle = lower factor
        return 0.8 + (0.4 * distance_from_middle)
    
    def _calculate_content_factor(self, segment: Dict[str, Any]) -> float:
        """
        Calculate content-based adjustment factor.
        
        Args:
            segment: The segment to score
            
        Returns:
            Content adjustment factor (0.7-1.3)
        """
        text = segment.get("text", "").lower()
        
        # Count high-salience keywords
        keyword_count = sum(1 for keyword in self.high_salience_keywords if keyword in text)
        
        # Check for numerical content (often important)
        has_numbers = bool(re.search(r'\d+', text))
        
        # Check for logical connectors
        has_logical_connectors = any(connector in text for connector in 
                                    ["if", "then", "because", "therefore", "thus"])
        
        # Base factor
        factor = 1.0
        
        # Adjust for keywords
        factor += min(0.3, keyword_count * 0.05)
        
        # Adjust for numbers
        if has_numbers:
            factor += 0.1
            
        # Adjust for logical connectors
        if has_logical_connectors:
            factor += 0.1
            
        # Cap the factor
        return min(1.3, factor)
    
    def _adjust_for_logical_flow(self, segments: List[Dict[str, Any]]) -> None:
        """
        Adjust scores to ensure logical flow is preserved.
        
        Args:
            segments: List of segments with initial scores
        """
        # Sort by position to ensure proper sequence
        segments.sort(key=lambda x: x.get("position", 0))
        
        # Ensure no adjacent low-salience segments (would break flow)
        for i in range(1, len(segments) - 1):
            prev_score = segments[i-1].get("salience_score", 0.5)
            curr_score = segments[i].get("salience_score", 0.5)
            next_score = segments[i+1].get("salience_score", 0.5)
            
            # If both neighbors are high-salience but current is low, boost it
            if prev_score > 0.7 and next_score > 0.7 and curr_score < 0.6:
                segments[i]["salience_score"] = 0.65
                logger.debug(f"Boosted segment {i} score for logical flow")
    
    def _score_with_bert(self, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Score segments using BERT-based classification.
        
        Args:
            segments: List of segment dictionaries
            
        Returns:
            List of segments with added salience_score field
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Prepare texts for batch processing
            texts = [segment.get("text", "") for segment in segments]
            
            # Get scores from model
            inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
            with torch.no_grad():
                outputs = self.model(**inputs)
                scores = torch.nn.functional.softmax(outputs.logits, dim=1)
                
                # Extract salience scores (assuming binary classification)
                salience_scores = scores[:, 1].tolist()
            
            # Add scores to segments
            for i, segment in enumerate(segments):
                segment["salience_score"] = salience_scores[i]
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in BERT scoring: {e}")
            logger.info("Falling back to rule-based scoring")
            return self._score_with_rules(segments)
    
    def _initialize_bert_model(self):
        """Initialize the BERT model for salience scoring."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # This would typically load a fine-tuned model for salience classification
            # For implementation, you would need to train this model on annotated reasoning traces
            model_name = "distilbert-base-uncased"  # Placeholder
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info(f"Loaded BERT model: {model_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import BERT dependencies: {e}")
            raise
