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
from typing import Dict, Any, List
from cryptotrading.analysis.retrieval import RetrievalEncoder
from cryptotrading.data.index import VectorIndex

logger = logging.getLogger(__name__)

class RetrievalServiceEncoder:
    """
    Encoder service for converting price segments and order book data into embedding vectors.
    
    Combines deep learning embeddings from the Embed Service with local handcrafted features.
    """
    
    def __init__(self, window_size: int = 60, n_fft: int = 32, dim: int = 184, embed_service_url: str = None):
        """
        Initialize the RetrievalServiceEncoder.

        Args:
            window_size (int): Size of the rolling price window. Defaults to 60.
            n_fft (int): Number of FFT bins for the handcrafted encoder. Defaults to 32.
            dim (int): Vector dimension of the index. Defaults to 184 (128 from embed + 56 local).
            embed_service_url (str, optional): Target URL for the embed service.
                If not specified, reads from the EMBED_SERVICE_URL environment variable,
                defaulting to 'http://localhost:8301'.
        """
        self.encoder = RetrievalEncoder(window_size=window_size, n_fft=n_fft)
        self.index = VectorIndex(dim=dim)
        self.dim = dim
        self.is_built = False
        self.embed_service_url = embed_service_url or os.getenv("EMBED_SERVICE_URL", "http://localhost:8301")

    def encode_segment(self, prices: np.ndarray, order_book: Dict[str, Any]) -> np.ndarray:
        """
        Encode a price segment and order book structure into a combined representation vector.

        This method generates a combined vector by concatenating the 128D deep learning
        representation from the Embed Service (retrieved over HTTP) with the 56D local
        handcrafted spectral and order book imbalance features, resulting in a 184D vector.

        If the requested index dimension is not 184, or if the HTTP call to the embed service
        fails, the method falls back to using the local handcrafted encoder output (padded
        or truncated to the target dimension).

        Args:
            prices (np.ndarray): 1D array of historical price returns.
            order_book (Dict[str, Any]): Dictionary containing order book 'bids' and 'asks'.

        Returns:
            np.ndarray: A 1D float32 array representing the segment embedding.
        """
        # Calculate local handcrafted representation (dimension 56)
        local_emb = self.encoder.encode(prices, order_book)
        
        # If the requested dimension is NOT the combined dimension of 184,
        # fall back to local handcrafted representation (padded or truncated to target dim)
        if self.dim != 184:
            if len(local_emb) < self.dim:
                return np.pad(local_emb, (0, self.dim - len(local_emb)), 'constant')
            elif len(local_emb) > self.dim:
                return local_emb[:self.dim]
            return local_emb
            
        try:
            payload = {"prices": prices.tolist()}
            response = httpx.post(f"{self.embed_service_url}/embed", json=payload, timeout=3.0)
            if response.status_code == 200:
                data = response.json()
                embed_emb = np.array(data["embedding"], dtype=np.float32)
                # Concatenate 128D embed representation + 56D local representation
                return np.concatenate([embed_emb, local_emb])
            else:
                logger.warning(f"Embed service returned {response.status_code}, falling back to padded local representation.")
        except (httpx.ConnectError, httpx.ConnectTimeout) as ce:
            logger.warning(f"Embed service connection failed: {ce}. Falling back to padded local representation.")
        except Exception as e:
            logger.error(f"Error encoding segment via embed service: {e}. Falling back to padded local representation.")
            
        # Padded local fallback to fit dim 184
        return np.pad(local_emb, (0, 184 - len(local_emb)), 'constant')

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

        if self.dim != 184:
            res = []
            for local_emb in local_embs:
                if len(local_emb) < self.dim:
                    res.append(np.pad(local_emb, (0, self.dim - len(local_emb)), 'constant'))
                elif len(local_emb) > self.dim:
                    res.append(local_emb[:self.dim])
                else:
                    res.append(local_emb)
            return res

        # Call the batch embed endpoint
        embeddings_128 = []
        batch_size = 500  # Batch size for HTTP requests

        try:
            for i in range(0, n, batch_size):
                sub_prices = [p.tolist() for p in prices_list[i:i+batch_size]]
                payload = {"prices_list": sub_prices}
                response = httpx.post(f"{self.embed_service_url}/embed/batch", json=payload, timeout=10.0)
                if response.status_code == 200:
                    data = response.json()
                    for emb in data["embeddings"]:
                        embeddings_128.append(np.array(emb, dtype=np.float32))
                else:
                    logger.warning(f"Embed service batch returned {response.status_code}, falling back to padded local representation for this batch.")
                    for j in range(i, min(i + batch_size, n)):
                        local_emb = local_embs[j]
                        embeddings_128.append(np.pad(local_emb, (0, 128), 'constant')[:128])
        except Exception as e:
            logger.error(f"Error encoding segments batch via embed service: {e}. Falling back to padded local representation.")
            while len(embeddings_128) < n:
                idx = len(embeddings_128)
                local_emb = local_embs[idx]
                embeddings_128.append(np.pad(local_emb, (0, 128), 'constant')[:128])

        # Concatenate 128D embed representation + 56D local representation
        combined_embs = []
        for i in range(n):
            combined_embs.append(np.concatenate([embeddings_128[i], local_embs[i]]))

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