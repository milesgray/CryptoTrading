"""Tests for the PostgreSQL adapter."""
import asyncio
import os
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

from cryptotrading.data.postgres import (
    init_db,
    close_db,
    price_repo,
    order_book_repo,
    document_embedding_repo,
    get_connection,
    transaction
)
from cryptotrading.config import POSTGRES_USE_PGVECTOR

# Test data
TEST_PRICE_DATA = {
    'time': datetime.utcnow(),
    'symbol': 'BTC/USDT',
    'exchange': 'test_exchange',
    'open': 40000.0,
    'high': 41000.0,
    'low': 39900.0,
    'close': 40500.0,
    'volume': 100.5,
    'metadata': {'test': True, 'source': 'test'}
}

TEST_ORDER_BOOK_DATA = {
    'time': datetime.utcnow(),
    'symbol': 'BTC/USDT',
    'exchange': 'test_exchange',
    'is_bid': True,
    'price': 40450.0,
    'amount': 1.5,
    'metadata': {'test': True}
}

TEST_EMBEDDING_DATA = {
    'id': 'test_doc_1',
    'content': 'This is a test document',
    'embedding': [0.1] * 1536,  # OpenAI embedding dimension
    'metadata': {'category': 'test', 'source': 'test'},
    'created_at': datetime.utcnow()
}

@pytest.fixture(scope="module")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="module")
async def db_setup():
    """Set up the test database."""
    await init_db()
    yield
    await close_db()

@pytest.mark.asyncio
async def test_connection(db_setup):
    """Test database connection."""
    async with get_connection() as conn:
        assert conn is not None
        version = await conn.fetchval('SELECT version()')
        assert 'PostgreSQL' in version

@pytest.mark.asyncio
async def test_transaction(db_setup):
    """Test transaction management."""
    async with transaction() as conn:
        result = await conn.fetchval('SELECT 1')
        assert result == 1

@pytest.mark.asyncio
async def test_price_repo_crud(db_setup):
    """Test CRUD operations for price data."""
    # Create
    inserted_id = await price_repo.insert(TEST_PRICE_DATA)
    assert inserted_id is not None
    
    # Read
    record = await price_repo.get(inserted_id)
    assert record is not None
    assert record['symbol'] == TEST_PRICE_DATA['symbol']
    
    # Update
    updated = await price_repo.update(
        inserted_id,
        {'close': 40600.0, 'volume': 150.75}
    )
    assert updated is True
    
    # Verify update
    updated_record = await price_repo.get(inserted_id)
    assert updated_record['close'] == 40600.0
    assert updated_record['volume'] == 150.75
    
    # Query
    records = await price_repo.query(
        filters={'symbol': 'BTC/USDT'},
        limit=10
    )
    assert len(records) > 0
    
    # Delete
    deleted = await price_repo.delete(inserted_id)
    assert deleted is True
    
    # Verify deletion
    deleted_record = await price_repo.get(inserted_id)
    assert deleted_record is None

@pytest.mark.asyncio
async def test_order_book_repo_crud(db_setup):
    """Test CRUD operations for order book data."""
    # Create
    inserted_id = await order_book_repo.insert(TEST_ORDER_BOOK_DATA)
    assert inserted_id is not None
    
    # Read
    record = await order_book_repo.get(inserted_id)
    assert record is not None
    assert record['symbol'] == TEST_ORDER_BOOK_DATA['symbol']
    assert record['is_bid'] == TEST_ORDER_BOOK_DATA['is_bid']
    
    # Query
    records = await order_book_repo.query(
        filters={
            'symbol': 'BTC/USDT',
            'exchange': 'test_exchange'
        },
        limit=10
    )
    assert len(records) > 0
    
    # Test order book snapshot
    snapshot = await order_book_repo.get_order_book_snapshot(
        symbol='BTC/USDT',
        exchange='test_exchange',
        depth=5
    )
    assert 'bids' in snapshot
    assert 'asks' in snapshot
    
    # Clean up
    await order_book_repo.delete(inserted_id)

@pytest.mark.skipif(not POSTGRES_USE_PGVECTOR, reason="pgvector not enabled")
@pytest.mark.asyncio
async def test_document_embedding_repo(db_setup):
    """Test document embedding operations."""
    if not document_embedding_repo:
        pytest.skip("Document embedding repository not initialized")
    
    # Create
    await document_embedding_repo.insert(TEST_EMBEDDING_DATA)
    
    # Similarity search
    similar = await document_embedding_repo.find_similar(
        embedding=TEST_EMBEDDING_DATA['embedding'],
        limit=1
    )
    
    assert len(similar) > 0
    assert similar[0]['id'] == TEST_EMBEDDING_DATA['id']
    assert 'similarity' in similar[0]
    
    # Clean up
    await document_embedding_repo.delete(TEST_EMBEDDING_DATA['id'])

@pytest.mark.asyncio
async def test_ohlcv_aggregation(db_setup):
    """Test OHLCV data aggregation."""
    # Insert test data
    now = datetime.utcnow()
    test_data = [
        {
            'time': now - timedelta(minutes=i*5),  # 5-minute intervals
            'symbol': 'TEST/USDT',
            'exchange': 'test_exchange',
            'open': 100.0 + i,
            'high': 101.0 + i,
            'low': 99.0 + i,
            'close': 100.5 + i,
            'volume': 10.0 + i,
            'metadata': {'test': True}
        }
        for i in range(12)  # 12 * 5min = 60min = 1h
    ]
    
    # Insert test data
    for data in test_data:
        await price_repo.insert(data)
    
    # Get 1h OHLCV
    ohlcv = await price_repo.get_ohlcv(
        symbol='TEST/USDT',
        timeframe='1h',
        exchange='test_exchange',
        limit=10
    )
    
    # Verify aggregation
    assert len(ohlcv) > 0
    assert 'open' in ohlcv[0]
    assert 'high' in ohlcv[0]
    assert 'low' in ohlcv[0]
    assert 'close' in ohlcv[0]
    assert 'volume' in ohlcv[0]
    
    # Clean up
    async with get_connection() as conn:
        await conn.execute("DELETE FROM price_data WHERE symbol = 'TEST/USDT'")

@pytest.mark.asyncio
async def test_concurrent_connections(db_setup):
    """Test handling of concurrent connections."""
    async def run_query(query_num):
        async with get_connection() as conn:
            result = await conn.fetchval('SELECT $1::text', f'test_{query_num}')
            return result
    
    # Run multiple queries concurrently
    tasks = [run_query(i) for i in range(10)]
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 10
    assert all(f'test_{i}' in results for i in range(10))
