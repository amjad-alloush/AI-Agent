from typing import List, Dict, Any, Optional, Tuple
import json
import os
import numpy as np
from datetime import datetime

# Import embedding model
from sentence_transformers import SentenceTransformer


class LongTermMemory:
    """
    Manages long-term memory for the agent using vector embeddings for
    semantic retrieval.
    """


def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
    # Initialize embedding model
    self.embedding_model = SentenceTransformer(embedding_model_name)
    # Initialize memory storage
    self.memories = []
    self.embeddings = None


def add_memory(self, text: str, metadata: Dict[str, Any] = None) -> int:
    """
    Add a new memory to long-term storage.
    Args:
    text: The text content to remember
    metadata: Additional information about the memory
    Returns:
    Index of the added memory
    """
    if metadata is None:
        metadata = {}

    # Create memory entry
    memory_entry = {
        "text": text,
        "timestamp": datetime.now().isoformat(),
        "metadata": metadata,
    }

    # Add to memories
    memory_index = len(self.memories)
    self.memories.append(memory_entry)
    # Update embeddings
    self._update_embeddings()
    return memory_index


def _update_embeddings(self) -> None:
    """Update the embedding matrix for all memories."""
    texts = [memory["text"] for memory in self.memories]
    self.embeddings = self.embedding_model.encode(texts)


def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for memories semantically related to the query.
    Args:
        query: The search query
        top_k: Number of results to return
    Returns:
        List of memory entries with similarity scores
    """
    if not self.memories:
        return []
    # Encode the query
    query_embedding = self.embedding_model.encode(query)
    # Calculate similarities
    similarities = np.dot(self.embeddings, query_embedding)
    # Get top k indices
    if len(similarities) < top_k:
        top_k = len(similarities)

    top_indices = np.argsort(similarities)[-top_k:][::-1]

    # Return results
    results = []
    for idx in top_indices:
        memory = self.memories[idx].copy()
        memory["similarity"] = float(similarities[idx])
        results.append(memory)

    return results


def get_memory_by_id(self, memory_id: int) -> Optional[Dict[str, Any]]:
    """Retrieve a specific memory by its ID."""
    if 0 <= memory_id < len(self.memories):
        return self.memories[memory_id]
    return None


def save_to_file(self, filepath: str) -> None:
    """Save memories to a file."""
    with open(filepath, "w") as f:
        json.dump(self.memories, f, indent=2)


def load_from_file(self, filepath: str) -> None:
    """Load memories from a file."""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            self.memories = json.load(f)
        self._update_embeddings()
