# services/retrieval/encoder.py

from cryptotrading.analysis.retrieval import RetrievalEncoder
from cryptotrading.data.index import VectorIndex
import numpy as np
from typing import Dict, Any, List

class RetrievalServiceEncoder:
    def __init__(self, window_size: int = 60, n_fft: int = 32, dim: int = 56):
        self.encoder = RetrievalEncoder(window_size=window_size, n_fft=n_fft)
        self.index = VectorIndex(dim=dim)
        self.is_built = False

    def encode_segment(self, prices: np.ndarray, order_book: Dict[str, Any]) -> np.ndarray:
        """Encode a price segment + order book into a vector."""
        return self.encoder.encode(prices, order_book)

    def add_segment(self, prices: np.ndarray, order_book: Dict[str, Any], metadata: Dict[str, Any]) -> int:
        """Add a segment to the index."""
        embedding = self.encode_segment(prices, order_book)
        return self.index.add_segment(embedding, metadata)

    def build_index(self, n_trees: int = 10) -> None:
        """Build the index for retrieval."""
        self.index.build(n_trees=n_trees)
        self.is_built = True

    def retrieve_segments(self, prices: np.ndarray, order_book: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k similar segments."""
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        query_embedding = self.encode_segment(prices, order_book)
        return self.index.retrieve(query_embedding, k=k)