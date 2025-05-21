"""
Consistency Validator module for Reasoning Trace Compression.

This module validates the logical consistency between original and compressed
reasoning traces to ensure that compression preserves key logical relationships.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re

logger = logging.getLogger(__name__)

class ConsistencyValidator:
    """
    Validates logical consistency between original and compressed reasoning traces.
    
    The ConsistencyValidator ensures that compression preserves the logical flow
    and key conclusions of the original reasoning trace.
    """
    
    def __init__(self, use_nli: bool = False, consistency_threshold: float = 0.7):
        """
        Initialize the ConsistencyValidator.
        
        Args:
            use_nli: Whether to use NLI models for validation (requires additional dependencies)
            consistency_threshold: Threshold score for considering traces consistent
        """
        self.use_nli = use_nli
        self.consistency_threshold = consistency_threshold
        
        # Initialize NLI model if requested
        if self.use_nli:
            try:
                self._initialize_nli_model()
            except ImportError:
                logger.warning("NLI dependencies not available. Falling back to rule-based validation.")
                self.use_nli = False
        
        logger.info(f"Initialized ConsistencyValidator (use_nli={use_nli})")
    
    def validate(self, original_trace: str, compressed_trace: str) -> Tuple[bool, float]:
        """
        Validate logical consistency between original and compressed traces.
        
        Args:
            original_trace: The original reasoning trace
            compressed_trace: The compressed reasoning trace
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        # Choose validation method based on configuration
        if self.use_nli:
            return self._validate_with_nli(original_trace, compressed_trace)
        else:
            return self._validate_with_rules(original_trace, compressed_trace)
    
    def _validate_with_rules(self, original_trace: str, compressed_trace: str) -> Tuple[bool, float]:
        """
        Validate consistency using rule-based heuristics.
        
        Args:
            original_trace: The original reasoning trace
            compressed_trace: The compressed reasoning trace
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        # Extract key elements from both traces
        original_elements = self._extract_key_elements(original_trace)
        compressed_elements = self._extract_key_elements(compressed_trace)
        
        # Calculate scores for different aspects of consistency
        conclusion_score = self._compare_conclusions(original_elements, compressed_elements)
        entity_score = self._compare_entities(original_elements, compressed_elements)
        relation_score = self._compare_relations(original_elements, compressed_elements)
        
        # Calculate overall consistency score
        consistency_score = 0.5 * conclusion_score + 0.3 * entity_score + 0.2 * relation_score
        
        # Determine if consistent based on threshold
        is_consistent = consistency_score >= self.consistency_threshold
        
        logger.debug(f"Consistency validation: score={consistency_score:.2f}, is_consistent={is_consistent}")
        return is_consistent, consistency_score
    
    def _extract_key_elements(self, trace: str) -> Dict[str, Any]:
        """
        Extract key elements from a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            Dictionary of extracted elements
        """
        # Extract conclusions
        conclusions = self._extract_conclusions(trace)
        
        # Extract entities (e.g., names, numbers, key terms)
        entities = self._extract_entities(trace)
        
        # Extract relations (e.g., causal relationships)
        relations = self._extract_relations(trace)
        
        return {
            "conclusions": conclusions,
            "entities": entities,
            "relations": relations
        }
    
    def _extract_conclusions(self, trace: str) -> List[str]:
        """
        Extract conclusions from a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            List of extracted conclusions
        """
        conclusions = []
        
        # Look for explicit conclusion markers
        conclusion_patterns = [
            r"(?i)conclusion:\s*(.*?)(?:\n|$)",
            r"(?i)therefore,\s*(.*?)(?:\n|$)",
            r"(?i)thus,\s*(.*?)(?:\n|$)",
            r"(?i)hence,\s*(.*?)(?:\n|$)",
            r"(?i)in conclusion,\s*(.*?)(?:\n|$)",
            r"(?i)to summarize,\s*(.*?)(?:\n|$)",
            r"(?i)finally,\s*(.*?)(?:\n|$)",
            r"(?i)the answer is\s*(.*?)(?:\n|$)"
        ]
        
        for pattern in conclusion_patterns:
            matches = re.findall(pattern, trace)
            conclusions.extend([match.strip() for match in matches if match.strip()])
        
        # If no explicit conclusions found, use the last sentence
        if not conclusions:
            sentences = self._split_into_sentences(trace)
            if sentences:
                conclusions.append(sentences[-1])
        
        return conclusions
    
    def _extract_entities(self, trace: str) -> List[str]:
        """
        Extract key entities from a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        # Extract numbers
        number_pattern = r'\b\d+(?:\.\d+)?\b'
        numbers = re.findall(number_pattern, trace)
        entities.extend(numbers)
        
        # Extract proper nouns (simplified approach)
        # In a real implementation, you might use NER models
        proper_noun_pattern = r'\b[A-Z][a-z]+\b'
        proper_nouns = re.findall(proper_noun_pattern, trace)
        entities.extend(proper_nouns)
        
        # Extract key terms (domain-specific)
        key_terms = ["hypothesis", "evidence", "experiment", "data", "model", "algorithm", "function"]
        for term in key_terms:
            if term in trace.lower():
                entities.append(term)
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_relations(self, trace: str) -> List[str]:
        """
        Extract key relations from a reasoning trace.
        
        Args:
            trace: Reasoning trace to analyze
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        # Look for causal relationships
        causal_patterns = [
            r"(?i)(.*?)\s+because\s+(.*?)(?:\.|$)",
            r"(?i)(.*?)\s+causes\s+(.*?)(?:\.|$)",
            r"(?i)(.*?)\s+results in\s+(.*?)(?:\.|$)",
            r"(?i)if\s+(.*?)\s+then\s+(.*?)(?:\.|$)"
        ]
        
        for pattern in causal_patterns:
            matches = re.findall(pattern, trace)
            relations.extend([f"{cause.strip()} -> {effect.strip()}" for cause, effect in matches])
        
        return relations
    
    def _compare_conclusions(self, original_elements: Dict[str, Any], compressed_elements: Dict[str, Any]) -> float:
        """
        Compare conclusions between original and compressed traces.
        
        Args:
            original_elements: Elements from original trace
            compressed_elements: Elements from compressed trace
            
        Returns:
            Similarity score (0.0-1.0)
        """
        original_conclusions = original_elements.get("conclusions", [])
        compressed_conclusions = compressed_elements.get("conclusions", [])
        
        if not original_conclusions or not compressed_conclusions:
            return 0.5  # Neutral score if no conclusions found
        
        # Calculate word overlap between conclusions
        total_similarity = 0.0
        
        for orig_concl in original_conclusions:
            orig_words = set(orig_concl.lower().split())
            
            # Find best matching conclusion in compressed trace
            best_similarity = 0.0
            for comp_concl in compressed_conclusions:
                comp_words = set(comp_concl.lower().split())
                
                if not orig_words or not comp_words:
                    continue
                
                # Jaccard similarity
                intersection = len(orig_words.intersection(comp_words))
                union = len(orig_words.union(comp_words))
                similarity = intersection / union if union > 0 else 0.0
                
                best_similarity = max(best_similarity, similarity)
            
            total_similarity += best_similarity
        
        # Average similarity across all original conclusions
        avg_similarity = total_similarity / len(original_conclusions) if original_conclusions else 0.0
        
        return avg_similarity
    
    def _compare_entities(self, original_elements: Dict[str, Any], compressed_elements: Dict[str, Any]) -> float:
        """
        Compare entities between original and compressed traces.
        
        Args:
            original_elements: Elements from original trace
            compressed_elements: Elements from compressed trace
            
        Returns:
            Similarity score (0.0-1.0)
        """
        original_entities = set(original_elements.get("entities", []))
        compressed_entities = set(compressed_elements.get("entities", []))
        
        if not original_entities:
            return 0.5  # Neutral score if no entities found
        
        # Calculate entity preservation ratio
        intersection = len(original_entities.intersection(compressed_entities))
        
        # Focus on preservation of original entities in compressed trace
        preservation_ratio = intersection / len(original_entities) if original_entities else 0.0
        
        return preservation_ratio
    
    def _compare_relations(self, original_elements: Dict[str, Any], compressed_elements: Dict[str, Any]) -> float:
        """
        Compare relations between original and compressed traces.
        
        Args:
            original_elements: Elements from original trace
            compressed_elements: Elements from compressed trace
            
        Returns:
            Similarity score (0.0-1.0)
        """
        original_relations = original_elements.get("relations", [])
        compressed_relations = compressed_elements.get("relations", [])
        
        if not original_relations:
            return 0.5  # Neutral score if no relations found
        
        # Calculate relation preservation ratio
        preserved_relations = 0
        
        for orig_rel in original_relations:
            # Check if any compressed relation is similar
            for comp_rel in compressed_relations:
                if self._are_relations_similar(orig_rel, comp_rel):
                    preserved_relations += 1
                    break
        
        preservation_ratio = preserved_relations / len(original_relations) if original_relations else 0.0
        
        return preservation_ratio
    
    def _are_relations_similar(self, relation1: str, relation2: str) -> bool:
        """
        Check if two relations are semantically similar.
        
        Args:
            relation1: First relation
            relation2: Second relation
            
        Returns:
            Boolean indicating similarity
        """
        # Simple word overlap approach
        words1 = set(relation1.lower().split())
        words2 = set(relation2.lower().split())
        
        if not words1 or not words2:
            return False
        
        # Calculate Jaccard similarity
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        similarity = intersection / union if union > 0 else 0.0
        
        # Consider similar if above threshold
        return similarity > 0.5
    
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
    
    def _validate_with_nli(self, original_trace: str, compressed_trace: str) -> Tuple[bool, float]:
        """
        Validate consistency using NLI models.
        
        Args:
            original_trace: The original reasoning trace
            compressed_trace: The compressed reasoning trace
            
        Returns:
            Tuple of (is_consistent, consistency_score)
        """
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Extract conclusions from both traces
            original_conclusions = self._extract_conclusions(original_trace)
            
            # If no explicit conclusions, use the whole trace
            if not original_conclusions:
                original_conclusions = [original_trace]
            
            # Calculate entailment scores
            entailment_scores = []
            
            for conclusion in original_conclusions:
                # Prepare inputs for the model
                inputs = self.tokenizer(
                    text=compressed_trace,
                    text_pair=conclusion,
                    padding=True,
                    truncation=True,
                    return_tensors="pt"
                )
                
                # Get predictions
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    predictions = torch.nn.functional.softmax(outputs.logits, dim=1)
                    
                    # Assuming standard NLI labels: [contradiction, neutral, entailment]
                    entailment_score = predictions[0, 2].item()
                    entailment_scores.append(entailment_score)
            
            # Average entailment score
            avg_score = sum(entailment_scores) / len(entailment_scores)
            
            # Determine if consistent based on threshold
            is_consistent = avg_score >= self.consistency_threshold
            
            logger.debug(f"NLI consistency validation: score={avg_score:.2f}, is_consistent={is_consistent}")
            return is_consistent, avg_score
            
        except Exception as e:
            logger.error(f"Error in NLI validation: {e}")
            logger.info("Falling back to rule-based validation")
            return self._validate_with_rules(original_trace, compressed_trace)
    
    def _initialize_nli_model(self):
        """Initialize the NLI model for consistency validation."""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            
            # Load pretrained NLI model
            model_name = "roberta-large-mnli"  # A common NLI model
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            logger.info(f"Loaded NLI model: {model_name}")
            
        except ImportError as e:
            logger.error(f"Failed to import NLI dependencies: {e}")
            raise
