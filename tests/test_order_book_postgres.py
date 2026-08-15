import pytest
import datetime as dt
from unittest.mock import AsyncMock, MagicMock, patch

from cryptotrading.data.postgres import OrderBookRepository
from cryptotrading.data.book import OrderBookPostgresAdapter
from cryptotrading.data.models import ExchangeRawOrderBook


@pytest.mark.asyncio
async def test_order_book_repo_store_order_book():
    repo = OrderBookRepository()
    mock_conn = AsyncMock()
    
    symbol = "BTC/USDT"
    exchange = "binance"
    now = dt.datetime(2026, 8, 14, 12, 0, 0, tzinfo=dt.timezone.utc)
    bids = [[100.0, 1.5], [99.0, 2.0], [100.0, 3.0]]  # duplicate price 100.0 should deduplicate
    asks = [[101.0, 1.2], [102.0, 2.5]]
    
    count = await repo.store_order_book(
        symbol=symbol,
        exchange=exchange,
        bids=bids,
        asks=asks,
        time=now,
        conn=mock_conn
    )
    
    assert count == 4  # 2 unique bids + 2 unique asks
    assert mock_conn.executemany.called
    query, records = mock_conn.executemany.call_args[0]
    assert "INSERT INTO order_book_data" in query
    assert "ON CONFLICT" in query
    assert len(records) == 4
    
    # Check records format: (time, symbol, exchange, is_bid, price, amount, metadata)
    bid_records = [r for r in records if r[3] is True]
    ask_records = [r for r in records if r[3] is False]
    assert len(bid_records) == 2
    assert len(ask_records) == 2
    assert (now, symbol, exchange, True, 100.0, 3.0, None) in bid_records


@pytest.mark.asyncio
async def test_order_book_repo_store_empty():
    repo = OrderBookRepository()
    mock_conn = AsyncMock()
    
    count = await repo.store_order_book(
        symbol="ETH/USDT",
        exchange="coinbase",
        bids=[],
        asks=[],
        conn=mock_conn
    )
    assert count == 0
    assert not mock_conn.executemany.called


@pytest.mark.asyncio
async def test_order_book_repo_get_order_book_snapshot():
    repo = OrderBookRepository()
    mock_conn = AsyncMock()
    
    now = dt.datetime(2026, 8, 14, 12, 0, 0, tzinfo=dt.timezone.utc)
    mock_conn.fetch.side_effect = [
        # bids response
        [{'price': 100.0, 'amount': 1.5}, {'price': 99.0, 'amount': 2.0}],
        # asks response
        [{'price': 101.0, 'amount': 1.2}, {'price': 102.0, 'amount': 2.5}],
    ]
    
    snapshot = await repo.get_order_book_snapshot(
        symbol="BTC/USDT",
        exchange="binance",
        time=now,
        depth=5,
        conn=mock_conn
    )
    
    assert 'bids' in snapshot and len(snapshot['bids']) == 2
    assert 'asks' in snapshot and len(snapshot['asks']) == 2
    assert snapshot['bids'][0] == {'price': 100.0, 'amount': 1.5}
    assert snapshot['asks'][0] == {'price': 101.0, 'amount': 1.2}
    assert mock_conn.fetch.call_count == 2


@pytest.mark.asyncio
async def test_order_book_repo_get_order_books():
    repo = OrderBookRepository()
    mock_conn = AsyncMock()
    
    t1 = dt.datetime(2026, 8, 14, 12, 0, 0, tzinfo=dt.timezone.utc)
    mock_conn.fetch.return_value = [
        {'time': t1, 'is_bid': True, 'price': 100.0, 'amount': 1.0},
        {'time': t1, 'is_bid': False, 'price': 101.0, 'amount': 1.5},
    ]
    
    snapshots = await repo.get_order_books(
        symbol="BTC/USDT",
        exchange="binance",
        start_time=t1 - dt.timedelta(minutes=5),
        end_time=t1,
        depth=10,
        conn=mock_conn
    )
    
    assert len(snapshots) == 1
    assert snapshots[0]['timestamp'] == t1
    assert snapshots[0]['bids'] == [(100.0, 1.0)]
    assert snapshots[0]['asks'] == [(101.0, 1.5)]


@pytest.mark.asyncio
async def test_adapter_store_exchange_order_book():
    adapter = OrderBookPostgresAdapter()
    
    raw_data = [
        {
            'exchange': 'binance',
            'bids': [(100.0, 1.0), (99.0, 2.0)],
            'asks': [(101.0, 1.0), (102.0, 2.0)],
        }
    ]
    
    mock_conn = AsyncMock()
    
    with patch('cryptotrading.data.book.get_connection') as mock_get_conn, \
         patch('cryptotrading.data.book.order_book_repo.store_order_book', new_callable=AsyncMock) as mock_store:
        mock_get_conn.return_value.__aenter__.return_value = mock_conn
        
        await adapter.store_exchange_order_book("BTC/USDT", raw_data)
        
        # Verify price_data row inserted
        assert mock_conn.execute.called
        price_data_query = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO price_data" in price_data_query
        
        # Verify order_book_repo.store_order_book called
        assert mock_store.called
        call_kwargs = mock_store.call_args[1]
        assert call_kwargs['symbol'] == "BTC/USDT"
        assert call_kwargs['exchange'] == "binance"
        assert call_kwargs['bids'] == [(100.0, 1.0), (99.0, 2.0)]
        assert call_kwargs['asks'] == [(101.0, 1.0), (102.0, 2.0)]


@pytest.mark.asyncio
async def test_adapter_store_composite_order_book_data():
    adapter = OrderBookPostgresAdapter()
    
    composite_book = {
        'bids': [(100.5, 5.0), (99.5, 10.0)],
        'asks': [(101.5, 4.0), (102.5, 8.0)],
    }
    raw_data = [{'exchange': 'binance'}, {'exchange': 'coinbase'}]
    
    mock_conn = AsyncMock()
    
    with patch('cryptotrading.data.book.get_connection') as mock_get_conn, \
         patch('cryptotrading.data.book.order_book_repo.store_order_book', new_callable=AsyncMock) as mock_store:
        mock_get_conn.return_value.__aenter__.return_value = mock_conn
        
        await adapter.store_composite_order_book_data("BTC/USDT", composite_book, raw_data)
        
        # Verify price_data composite inserted
        assert mock_conn.execute.called
        price_data_query = mock_conn.execute.call_args[0][0]
        assert "INSERT INTO price_data" in price_data_query
        assert mock_conn.execute.call_args[0][3] == 'composite'
        
        # Verify order_book_repo.store_order_book called with exchange='composite'
        assert mock_store.called
        call_kwargs = mock_store.call_args[1]
        assert call_kwargs['symbol'] == "BTC/USDT"
        assert call_kwargs['exchange'] == "composite"
        assert call_kwargs['bids'] == [(100.5, 5.0), (99.5, 10.0)]
        assert call_kwargs['asks'] == [(101.5, 4.0), (102.5, 8.0)]
