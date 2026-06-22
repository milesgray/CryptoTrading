from annoy import AnnoyIndex
import numpy as np
from typing import List, Dict, Any

class VectorIndex:
    def __init__(self, dim: int = 100, metric: str = "euclidean"):
        self.index = AnnoyIndex(dim, metric)
        self.segments = []  # Store raw segments for retrieval
        self.dim = dim

    def add_segment(self, embedding: np.ndarray, segment: Dict[str, Any]) -> int:
        """Add a segment to the index."""
        if len(embedding) != self.dim:
            raise ValueError(f"Embedding dimension {len(embedding)} != index dimension {self.dim}")
        idx = len(self.segments)
        self.index.add_item(idx, embedding)
        self.segments.append(segment)
        return idx

    def build(self, n_trees: int = 10) -> None:
        """Build the index."""
        self.index.build(n_trees)

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k similar segments."""
        if len(query_embedding) != self.dim:
            raise ValueError(f"Query embedding dimension {len(query_embedding)} != index dimension {self.dim}")
        indices = self.index.get_nns_by_vector(query_embedding, k)
        return [self.segments[i] for i in indices]