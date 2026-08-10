# services/retrieval/encoder.py

"""
Retrieval Service Encoder Module.

This module provides the RetrievalServiceEncoder class, which wraps the pattern matching
representation logic. It combines the 128-dimensional deep learning embeddings from the
Embed Service with local handcrafted 56-dimensional spectral/orderbook features into a
combined 184-dimensional representation vector.
"""

import os
import httpx
import logging
import numpy as np
from typing import Dict, Any, List, Optional
from cryptotrading.analysis.retrieval import RetrievalEncoder
from cryptotrading.data.index import VectorIndex

logger = logging.getLogger(__name__)

class RetrievalServiceEncoder:
    """
    Encoder service for converting price segments and order book data into embedding vectors.
    
    Combines deep learning embeddings from the Embed Service with local handcrafted features.
    """
    
    def __init__(
        self,
        forecast_size: int = 15,
        historic_window_size: Optional[int] = None,
        n_fft: int = 32,
        dim: int = 237,
        embed_service_url: str = None,
        embed_dim: int = None
    ):
        """
        Initialize the RetrievalServiceEncoder.

        Args:
            forecast_size (int): Size of prediction horizon window. Defaults to 15.
            historic_window_size (int, optional): Size of historic price window. Defaults to 4 * forecast_size.
            n_fft (int): Number of FFT bins for handcrafted encoder. Defaults to 32.
            dim (int): Vector dimension of index. Defaults to 237.
            embed_service_url (str, optional): URL for embed service.
            embed_dim (int, optional): Embedding dimension from embed service.
        """
        self.forecast_size = forecast_size
        self.historic_window_size = historic_window_size if historic_window_size is not None else 4 * forecast_size
        self.encoder = RetrievalEncoder(
            forecast_size=self.forecast_size,
            historic_window_size=self.historic_window_size,
            n_fft=n_fft
        )
        self.index = VectorIndex(dim=dim)
        self.dim = dim
        self.is_built = False
        self.embed_service_url = embed_service_url or os.getenv("EMBED_SERVICE_URL", "http://embed:8301")
        
        # Determine deep learning embedding dimension dynamically if not provided
        if embed_dim is not None:
            self.embed_dim = embed_dim
            logger.info(f"RetrievalServiceEncoder initialized with embed_dim: {self.embed_dim}")
        else:
            self.embed_dim = 128
            import time
            for attempt in range(5):
                try:
                    response = httpx.get(f"{self.embed_service_url}/health", timeout=5.0)
                    if response.status_code == 200:
                        self.embed_dim = response.json().get("embedding_dim", 128)
                        logger.info(f"Dynamically determined embed service embedding dimension: {self.embed_dim}")
                        break
                except Exception as e:
                    logger.warning(f"Could not connect to embed service to determine dimension (attempt {attempt+1}/5): {e}.")
                    if attempt < 4:
                        time.sleep(2.0)

    def encode_segment(self, prices: np.ndarray, order_book: Dict[str, Any]) -> np.ndarray:
        """
        Encode a price segment and order book structure into a combined representation vector.

        This method generates a combined vector by concatenating the self.embed_dim deep learning
        representation from the Embed Service (retrieved over HTTP) with the local
        handcrafted spectral and order book imbalance features, resulting in a combined vector.

        If the requested index dimension is not expected, or if the HTTP call to the embed service
        fails, the method falls back to using the local handcrafted encoder output (padded
        or truncated to the target dimension).

        Args:
            prices (np.ndarray): 1D array of historical price returns.
            order_book (Dict[str, Any]): Dictionary containing order book 'bids' and 'asks'.

        Returns:
            np.ndarray: A 1D float32 array representing the segment embedding.
        """
        # Calculate local handcrafted representation
        local_emb = self.encoder.encode(prices, order_book)
        local_dim = len(local_emb)
        expected_combined_dim = self.embed_dim + local_dim
        
        # If the requested dimension is NOT the combined dimension,
        # fall back to local handcrafted representation (padded or truncated to target dim)
        if self.dim != expected_combined_dim:
            if len(local_emb) < self.dim:
                return np.pad(local_emb, (0, self.dim - len(local_emb)), 'constant')
            elif len(local_emb) > self.dim:
                return local_emb[:self.dim]
            return local_emb
            
        try:
            from cryptotrading.client import EmbedServiceClient
            client = EmbedServiceClient(base_url=self.embed_service_url)
            res = client.generate_embedding(prices.tolist())
            embed_emb = np.array(res["embedding"], dtype=np.float32)
            
            # Check dynamic dimension match
            if len(embed_emb) != self.embed_dim:
                self.embed_dim = len(embed_emb)
                
            combined = np.concatenate([embed_emb, local_emb])
            if len(combined) != self.dim:
                if len(combined) < self.dim:
                    return np.pad(combined, (0, self.dim - len(combined)), 'constant')
                return combined[:self.dim]
            return combined
        except Exception as e:
            logger.error(f"Error encoding segment via embed service: {e}. Falling back to padded local representation.")
            
        # Padded local fallback to fit self.dim
        if len(local_emb) < self.dim:
            return np.pad(local_emb, (0, self.dim - len(local_emb)), 'constant')
        elif len(local_emb) > self.dim:
            return local_emb[:self.dim]
        return local_emb

    def encode_segments_batch(self, prices_list: List[np.ndarray], order_books: List[Dict[str, Any]]) -> List[np.ndarray]:
        """
        Batch encode price segments and order book structures.

        Args:
            prices_list (List[np.ndarray]): List of 1D arrays of price returns.
            order_books (List[Dict[str, Any]]): List of order book dictionaries.

        Returns:
            List[np.ndarray]: List of 1D float32 arrays representing segment embeddings.
        """
        n = len(prices_list)
        if n == 0:
            return []

        # Calculate local handcrafted representations
        local_embs = [self.encoder.encode(prices_list[i], order_books[i]) for i in range(n)]
        local_dim = len(local_embs[0]) if n > 0 else 0
        expected_combined_dim = self.embed_dim + local_dim

        # Call the batch embed endpoint
        embeddings_embed = []
        batch_size = 100  # Batch size for HTTP requests

        try:
            for i in range(0, n, batch_size):
                sub_prices = [p.tolist() for p in prices_list[i:i+batch_size]]
                payload = {"prices_list": sub_prices}
                response = httpx.post(f"{self.embed_service_url}/embed/batch", json=payload, timeout=30.0)
                if response.status_code == 200:
                    data = response.json()
                    embs = data.get("embeddings", [])
                    if len(embs) == len(sub_prices):
                        for emb in embs:
                            emb_arr = np.array(emb, dtype=np.float32)
                            if len(emb_arr) != self.embed_dim:
                                self.embed_dim = len(emb_arr)
                            embeddings_embed.append(emb_arr)
                    else:
                        logger.warning(f"Embed service returned {len(embs)} embeddings for {len(sub_prices)} requests. Using fallback.")
                        for _ in range(len(sub_prices)):
                            embeddings_embed.append(np.zeros(self.embed_dim, dtype=np.float32))
                else:
                    logger.warning(f"Embed service batch returned {response.status_code}, falling back to padded local representation for this batch.")
                    for _ in range(i, min(i + batch_size, n)):
                        embeddings_embed.append(np.zeros(self.embed_dim, dtype=np.float32))
        except Exception as e:
            logger.error(f"Error encoding segments batch via embed service: {e}. Falling back to padded local representation.")
            while len(embeddings_embed) < n:
                embeddings_embed.append(np.zeros(self.embed_dim, dtype=np.float32))

        # Concatenate embed representation + local representation
        combined_embs = []
        for i in range(n):
            comb = np.concatenate([embeddings_embed[i], local_embs[i]])
            if len(comb) != self.dim:
                if len(comb) < self.dim:
                    comb = np.pad(comb, (0, self.dim - len(comb)), 'constant')
                else:
                    comb = comb[:self.dim]
            combined_embs.append(comb)

        return combined_embs

    def add_segment(self, prices: np.ndarray, order_book: Dict[str, Any], metadata: Dict[str, Any]) -> int:
        """
        Encode and index a historical price segment.

        Args:
            prices (np.ndarray): 1D array of historical price returns.
            order_book (Dict[str, Any]): Dictionary containing bids/asks order book structure.
            metadata (Dict[str, Any]): Associated reference payload (segment ID, prices list, etc.).

        Returns:
            int: The index position of the added segment in the vector database.
        """
        embedding = self.encode_segment(prices, order_book)
        return self.index.add_segment(embedding, metadata)

    def add_segments_batch(self, prices_list: List[np.ndarray], order_books: List[Dict[str, Any]], metadatas: List[Dict[str, Any]]) -> List[int]:
        """
        Encode and index a batch of historical price segments.

        Args:
            prices_list (List[np.ndarray]): List of 1D arrays of price returns.
            order_books (List[Dict[str, Any]]): List of order book dictionaries.
            metadatas (List[Dict[str, Any]]): List of metadata dictionaries.

        Returns:
            List[int]: List of index positions of the added segments in the vector database.
        """
        embeddings = self.encode_segments_batch(prices_list, order_books)
        ids = []
        for emb, meta in zip(embeddings, metadatas):
            idx_pos = self.index.add_segment(emb, meta)
            ids.append(idx_pos)
        return ids

    def build_index(self, n_trees: int = 10) -> None:
        """
        Build the Annoy search index to enable retrieval.

        Args:
            n_trees (int): Number of trees for building the Annoy index. Defaults to 10.
        """
        self.index.build(n_trees=n_trees)
        self.is_built = True

    def retrieve_segments(self, prices: np.ndarray, order_book: Dict[str, Any], k: int = 5) -> List[Dict[str, Any]]:
        """
        Query the vector database for the top-k most similar historical segments.

        Args:
            prices (np.ndarray): 1D array of query price returns.
            order_book (Dict[str, Any]): Dictionary containing query order book structure.
            k (int): Number of nearest segments to retrieve. Defaults to 5.

        Returns:
            List[Dict[str, Any]]: A list of metadata dictionaries corresponding to the matched segments.

        Raises:
            RuntimeError: If the index has not been built yet.
        """
        if not self.is_built:
            raise RuntimeError("Index not built. Call build_index() first.")
        query_embedding = self.encode_segment(prices, order_book)
        return self.index.retrieve(query_embedding, k=k)