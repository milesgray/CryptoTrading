"""
Simple NumPy + FAISS vector store for trade setup embeddings.

Stores everything in .npz files - no database infrastructure needed.
Uses FAISS for efficient similarity search.
"""

import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
import json
import logging

logger = logging.getLogger(__name__)

# Try to import FAISS, fall back to brute force if not available
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logger.warning("FAISS not installed - using brute force search (slower)")


@dataclass
class StoredTradeSetup:
    """A trade setup stored in the vector store"""
    id: int
    direction: int  # 1=long, -1=short
    profit_pct: float
    leverage: float
    hold_duration: int
    entry_timestamp: float
    entry_price: float
    exit_price: float
    symbol: str
    timeframe: str
    window_size: int


@dataclass 
class SimilarSetup:
    """Result from similarity search"""
    setup: StoredTradeSetup
    similarity: float
    distance: float


class NumpyVectorStore:
    """
    Simple vector store using NumPy arrays and FAISS.
    
    Storage format:
    - embeddings.npy: (N, embedding_dim) float32 array
    - metadata.json: List of setup metadata dicts
    - price_windows.npy: (N, window_size) float32 array (optional)
    - index.faiss: FAISS index file (if FAISS available)
    """
    
    def __init__(
        self,
        store_path: str = "vector_store",
        embedding_dim: int = 128,
        use_faiss: bool = True
    ):
        self.store_path = Path(store_path)
        self.embedding_dim = embedding_dim
        self.use_faiss = use_faiss and FAISS_AVAILABLE
        
        # In-memory data
        self.embeddings: Optional[np.ndarray] = None  # (N, embedding_dim)
        self.metadata: List[Dict] = []
        self.price_windows: Optional[np.ndarray] = None  # (N, window_size)
        self.faiss_index = None
        
        # Load existing data if available
        if self.store_path.exists():
            self.load()
    
    def _build_faiss_index(self):
        """Build or rebuild FAISS index from embeddings"""
        if not self.use_faiss or self.embeddings is None or len(self.embeddings) == 0:
            self.faiss_index = None
            return
        
        # Use IndexFlatIP for cosine similarity (embeddings should be L2 normalized)
        self.faiss_index = faiss.IndexFlatIP(self.embedding_dim)
        
        # Normalize embeddings for cosine similarity
        normalized = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8)
        self.faiss_index.add(normalized.astype(np.float32))
        
        logger.info(f"Built FAISS index with {self.faiss_index.ntotal} vectors")
    
    def add(
        self,
        embedding: np.ndarray,
        setup: StoredTradeSetup,
        price_window: Optional[np.ndarray] = None
    ) -> int:
        """
        Add a single setup to the store.
        
        Args:
            embedding: (embedding_dim,) array
            setup: Setup metadata
            price_window: Optional raw price window for visualization
            
        Returns:
            ID of the added setup
        """
        # Assign ID
        setup.id = len(self.metadata)
        
        # Add embedding
        embedding = embedding.reshape(1, -1).astype(np.float32)
        if self.embeddings is None:
            self.embeddings = embedding
        else:
            self.embeddings = np.vstack([self.embeddings, embedding])
        
        # Add metadata
        self.metadata.append(asdict(setup))
        
        # Add price window
        if price_window is not None:
            price_window = price_window.reshape(1, -1).astype(np.float32)
            if self.price_windows is None:
                self.price_windows = price_window
            else:
                # Pad if different lengths
                if price_window.shape[1] != self.price_windows.shape[1]:
                    max_len = max(price_window.shape[1], self.price_windows.shape[1])
                    if price_window.shape[1] < max_len:
                        price_window = np.pad(price_window, ((0, 0), (max_len - price_window.shape[1], 0)))
                    if self.price_windows.shape[1] < max_len:
                        self.price_windows = np.pad(self.price_windows, ((0, 0), (max_len - self.price_windows.shape[1], 0)))
                self.price_windows = np.vstack([self.price_windows, price_window])
        
        # Rebuild index periodically (every 1000 additions)
        if self.use_faiss and len(self.metadata) % 1000 == 0:
            self._build_faiss_index()
        
        return setup.id
    
    def add_batch(
        self,
        embeddings: np.ndarray,
        setups: List[StoredTradeSetup],
        price_windows: Optional[np.ndarray] = None
    ) -> List[int]:
        """
        Add multiple setups at once (more efficient).
        
        Args:
            embeddings: (N, embedding_dim) array
            setups: List of setup metadata
            price_windows: Optional (N, window_size) array
            
        Returns:
            List of IDs
        """
        start_id = len(self.metadata)
        ids = []
        
        # Assign IDs
        for i, setup in enumerate(setups):
            setup.id = start_id + i
            ids.append(setup.id)
        
        # Add embeddings
        embeddings = embeddings.astype(np.float32)
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Add metadata
        self.metadata.extend([asdict(s) for s in setups])
        
        # Add price windows
        if price_windows is not None:
            price_windows = price_windows.astype(np.float32)
            if self.price_windows is None:
                self.price_windows = price_windows
            else:
                # Handle different window sizes
                if price_windows.shape[1] != self.price_windows.shape[1]:
                    max_len = max(price_windows.shape[1], self.price_windows.shape[1])
                    if price_windows.shape[1] < max_len:
                        price_windows = np.pad(price_windows, ((0, 0), (max_len - price_windows.shape[1], 0)))
                    if self.price_windows.shape[1] < max_len:
                        self.price_windows = np.pad(self.price_windows, ((0, 0), (max_len - self.price_windows.shape[1], 0)))
                self.price_windows = np.vstack([self.price_windows, price_windows])
        
        # Build index
        self._build_faiss_index()
        
        logger.info(f"Added {len(setups)} setups to store (total: {len(self.metadata)})")
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 10,
        symbol: Optional[str] = None,
        direction: Optional[int] = None,
        min_profit: Optional[float] = None,
        max_profit: Optional[float] = None
    ) -> List[SimilarSetup]:
        """
        Find k most similar setups to query.
        
        Args:
            query_embedding: (embedding_dim,) query vector
            k: Number of results
            symbol: Filter by symbol
            direction: Filter by direction (1=long, -1=short)
            min_profit: Minimum profit filter
            max_profit: Maximum profit filter
            
        Returns:
            List of SimilarSetup objects sorted by similarity (descending)
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            return []
        
        query = query_embedding.reshape(1, -1).astype(np.float32)
        
        # Normalize query for cosine similarity
        query = query / (np.linalg.norm(query) + 1e-8)
        
        # Build filter mask
        mask = np.ones(len(self.metadata), dtype=bool)
        
        if symbol is not None:
            mask &= np.array([m['symbol'] == symbol for m in self.metadata])
        if direction is not None:
            mask &= np.array([m['direction'] == direction for m in self.metadata])
        if min_profit is not None:
            mask &= np.array([m['profit_pct'] >= min_profit for m in self.metadata])
        if max_profit is not None:
            mask &= np.array([m['profit_pct'] <= max_profit for m in self.metadata])
        
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Search
        if self.use_faiss and self.faiss_index is not None and not any([symbol, direction, min_profit, max_profit]):
            # Use FAISS for unfiltered search
            similarities, indices = self.faiss_index.search(query, min(k, self.faiss_index.ntotal))
            similarities = similarities[0]
            indices = indices[0]
        else:
            # Brute force search (with filtering)
            normalized_embeddings = self.embeddings[valid_indices] / (
                np.linalg.norm(self.embeddings[valid_indices], axis=1, keepdims=True) + 1e-8
            )
            similarities = (query @ normalized_embeddings.T)[0]
            
            # Get top k
            top_k_local = np.argsort(similarities)[::-1][:k]
            indices = valid_indices[top_k_local]
            similarities = similarities[top_k_local]
        
        # Build results
        results = []
        for idx, sim in zip(indices, similarities):
            if idx < 0:  # FAISS returns -1 for empty slots
                continue
                
            meta = self.metadata[idx]
            setup = StoredTradeSetup(**meta)
            
            results.append(SimilarSetup(
                setup=setup,
                similarity=float(sim),
                distance=float(1 - sim)
            ))
        
        return results
    
    def get_by_id(self, setup_id: int) -> Optional[Tuple[StoredTradeSetup, np.ndarray, Optional[np.ndarray]]]:
        """
        Get a setup by ID.
        
        Returns:
            Tuple of (setup, embedding, price_window) or None
        """
        if setup_id < 0 or setup_id >= len(self.metadata):
            return None
        
        setup = StoredTradeSetup(**self.metadata[setup_id])
        embedding = self.embeddings[setup_id]
        price_window = self.price_windows[setup_id] if self.price_windows is not None else None
        
        return setup, embedding, price_window
    
    def get_price_window(self, setup_id: int) -> Optional[np.ndarray]:
        """Get price window for a setup"""
        if self.price_windows is None or setup_id < 0 or setup_id >= len(self.price_windows):
            return None
        return self.price_windows[setup_id]
    
    def save(self):
        """Save store to disk"""
        self.store_path.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings
        if self.embeddings is not None:
            np.save(self.store_path / "embeddings.npy", self.embeddings)
        
        # Save metadata
        with open(self.store_path / "metadata.json", 'w') as f:
            json.dump(self.metadata, f)
        
        # Save price windows
        if self.price_windows is not None:
            np.save(self.store_path / "price_windows.npy", self.price_windows)
        
        # Save FAISS index
        if self.use_faiss and self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(self.store_path / "index.faiss"))
        
        logger.info(f"Saved vector store to {self.store_path} ({len(self.metadata)} setups)")
    
    def load(self):
        """Load store from disk"""
        if not self.store_path.exists():
            logger.warning(f"Store path {self.store_path} does not exist")
            return
        
        # Load embeddings
        emb_path = self.store_path / "embeddings.npy"
        if emb_path.exists():
            self.embeddings = np.load(emb_path)
            self.embedding_dim = self.embeddings.shape[1]
        
        # Load metadata
        meta_path = self.store_path / "metadata.json"
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                self.metadata = json.load(f)
        
        # Load price windows
        pw_path = self.store_path / "price_windows.npy"
        if pw_path.exists():
            self.price_windows = np.load(pw_path)
        
        # Load or rebuild FAISS index
        faiss_path = self.store_path / "index.faiss"
        if self.use_faiss and faiss_path.exists():
            self.faiss_index = faiss.read_index(str(faiss_path))
            logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
        elif self.use_faiss:
            self._build_faiss_index()
        
        logger.info(f"Loaded vector store from {self.store_path} ({len(self.metadata)} setups)")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get store statistics"""
        if not self.metadata:
            return {'total_setups': 0}
        
        profits = [m['profit_pct'] for m in self.metadata]
        directions = [m['direction'] for m in self.metadata]
        symbols = [m['symbol'] for m in self.metadata]
        
        from collections import Counter
        
        return {
            'total_setups': len(self.metadata),
            'by_symbol': dict(Counter(symbols)),
            'by_direction': {
                'long': sum(1 for d in directions if d == 1),
                'short': sum(1 for d in directions if d == -1)
            },
            'profit_stats': {
                'mean': float(np.mean(profits)),
                'std': float(np.std(profits)),
                'min': float(np.min(profits)),
                'max': float(np.max(profits)),
                'median': float(np.median(profits))
            },
            'embedding_dim': self.embedding_dim,
            'has_price_windows': self.price_windows is not None,
            'faiss_enabled': self.use_faiss and self.faiss_index is not None
        }
    
    def clear(self):
        """Clear all data"""
        self.embeddings = None
        self.metadata = []
        self.price_windows = None
        self.faiss_index = None
        logger.warning("Cleared all data from vector store")


# Convenience functions for the API
class VectorStoreManager:
    """Singleton manager for the vector store"""
    
    _instance: Optional[NumpyVectorStore] = None
    
    @classmethod
    def get_store(cls, store_path: str = "vector_store", **kwargs) -> NumpyVectorStore:
        if cls._instance is None:
            cls._instance = NumpyVectorStore(store_path=store_path, **kwargs)
        return cls._instance
    
    @classmethod
    def reset(cls):
        cls._instance = None
