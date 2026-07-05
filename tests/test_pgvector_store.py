import sys
import numpy as np
import pytest
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from cryptotrading.data.pgvector_store import (
    StoredTradeSetup,
    SimilarSetup,
    TradeEmbeddingDB,
    TradeEmbeddingDBSync
)

# Test forwarder compatibility
from services.embed.database.pgvector_store import (
    StoredTradeSetup as ForwardedStoredTradeSetup,
    SimilarSetup as ForwardedSimilarSetup,
    TradeEmbeddingDB as ForwardedTradeEmbeddingDB,
    TradeEmbeddingDBSync as ForwardedTradeEmbeddingDBSync
)

def test_imports_forwarder_compatibility():
    """Verify forwarder module correctly exposes classes from the main library."""
    assert StoredTradeSetup is ForwardedStoredTradeSetup
    assert SimilarSetup is ForwardedSimilarSetup
    assert TradeEmbeddingDB is ForwardedTradeEmbeddingDB
    assert TradeEmbeddingDBSync is ForwardedTradeEmbeddingDBSync


def test_stored_trade_setup_dataclass():
    """Verify StoredTradeSetup dataclass instantiation and default values."""
    setup = StoredTradeSetup(
        id=42,
        embedding=np.zeros(128, dtype=np.float32),
        direction=1,
        profit_pct=0.05,
        leverage=10.0,
        hold_duration=60,
        entry_timestamp=1719918000.0,
        entry_price=50000.0,
        exit_price=52500.0,
        symbol="BTC/USDT",
        timeframe="1m",
        window_size=100
    )
    
    assert setup.id == 42
    assert setup.direction == 1
    assert setup.profit_pct == 0.05
    assert setup.leverage == 10.0
    assert setup.price_window is None
    assert setup.created_at is None


@pytest.mark.asyncio
async def test_db_pool_fallback_connection():
    """Verify that TradeEmbeddingDB defaults to shared connection pool in connect()."""
    db = TradeEmbeddingDB(embedding_dim=128)
    
    mock_pool = MagicMock()
    
    with patch("cryptotrading.data.postgres._pool", mock_pool):
        await db.connect()
        assert db.pool is mock_pool


@pytest.mark.asyncio
async def test_db_pool_dedicated_dsn_connection():
    """Verify that TradeEmbeddingDB connects to a dedicated pool if DSN is passed."""
    db = TradeEmbeddingDB(dsn="postgresql://user:pass@host:5432/db", embedding_dim=128)
    
    mock_create_pool = AsyncMock()
    
    with patch("cryptotrading.data.postgres._pool", None):
        with patch("asyncpg.create_pool", mock_create_pool):
            await db.connect()
            mock_create_pool.assert_called_once_with(
                dsn="postgresql://user:pass@host:5432/db",
                min_size=2,
                max_size=10
            )
            assert db.pool is mock_create_pool.return_value


@pytest.mark.asyncio
async def test_db_initialize_schema():
    """Verify initialize_schema executes the table and HNSW index setup queries."""
    db = TradeEmbeddingDB(embedding_dim=128)
    db.pool = MagicMock()
    
    mock_conn = AsyncMock()
    
    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn
        
    db.pool.acquire = mock_acquire
    
    await db.initialize_schema()
    
    assert mock_conn.execute.call_count >= 3
    mock_conn.execute.assert_any_call("CREATE EXTENSION IF NOT EXISTS vector")


@pytest.mark.asyncio
async def test_db_insert_setup():
    """Verify insert_setup formats and executes the insert query correctly."""
    db = TradeEmbeddingDB(embedding_dim=128)
    db.pool = MagicMock()
    
    mock_conn = AsyncMock()
    
    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn
        
    db.pool.acquire = mock_acquire
    mock_conn.fetchrow.return_value = {"id": 100}
    
    setup = StoredTradeSetup(
        id=None,
        embedding=np.ones(128, dtype=np.float32),
        direction=-1,
        profit_pct=-0.02,
        leverage=5.0,
        hold_duration=30,
        entry_timestamp=1719918000.0,
        entry_price=100.0,
        exit_price=98.0,
        symbol="ETH/USDT",
        timeframe="1s",
        window_size=50,
        price_window=np.array([101.0, 100.0], dtype=np.float32)
    )
    
    setup_id = await db.insert_setup(setup)
    
    assert setup_id == 100
    mock_conn.fetchrow.assert_called_once()
    args = mock_conn.fetchrow.call_args[0]
    
    assert "INSERT INTO trade_setups" in args[0]
    assert args[1].startswith("[1.0,1.0")  # serialized vector
    assert args[2] == -1  # direction
    assert args[3] == -0.02  # profit
    assert args[9] == "ETH/USDT"  # symbol
    assert args[10] == "1s"  # timeframe


@pytest.mark.asyncio
async def test_db_search_similar():
    """Verify search_similar builds filters and returns SimilarSetup dataclasses."""
    db = TradeEmbeddingDB(embedding_dim=128)
    db.pool = MagicMock()
    
    mock_conn = AsyncMock()
    
    @asynccontextmanager
    async def mock_acquire():
        yield mock_conn
        
    db.pool.acquire = mock_acquire
    
    mock_row = {
        "id": 1,
        "embedding": "[0.1,0.2]",
        "direction": 1,
        "profit_pct": 0.05,
        "leverage": 10.0,
        "hold_duration": 60,
        "entry_timestamp": 1719918000.0,
        "entry_price": 50000.0,
        "exit_price": 52500.0,
        "symbol": "BTC/USDT",
        "timeframe": "1m",
        "window_size": 100,
        "price_window": np.array([49000.0, 50000.0], dtype=np.float32).tobytes(),
        "created_at": datetime.now(),
        "similarity": 0.95,
        "l2_distance": 0.05
    }
    mock_conn.fetch.return_value = [mock_row]
    
    query_emb = np.ones(128, dtype=np.float32)
    results = await db.search_similar(
        query_embedding=query_emb,
        k=1,
        symbol="BTC/USDT",
        direction=1
    )
    
    assert len(results) == 1
    similar = results[0]
    assert isinstance(similar, SimilarSetup)
    assert similar.similarity == 0.95
    assert similar.distance == 0.05
    assert similar.setup.id == 1
    assert similar.setup.symbol == "BTC/USDT"
    assert similar.setup.direction == 1
    assert np.allclose(similar.setup.price_window, np.array([49000.0, 50000.0]))
