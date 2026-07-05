"""
Backward-compatible forwarder for Trade Setup pgvector store.
Imports all core database adapter logic from the main cryptotrading package.
"""

from cryptotrading.data.pgvector_store import (
    StoredTradeSetup,
    SimilarSetup,
    TradeEmbeddingDB,
    TradeEmbeddingDBSync
)
