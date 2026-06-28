import pytest
import datetime as dt
import numpy as np
import json
import threading
from unittest.mock import MagicMock, patch
from pathlib import Path

from cryptotrading.trade.price.exchange import ExchangePriceClient, _get_file_cache_path

@pytest.fixture
def mock_exchange():
    with patch('ccxt.binanceus') as mock_ccxt_class:
        mock_instance = MagicMock()
        mock_ccxt_class.return_value = mock_instance
        mock_instance.rateLimit = 50
        mock_instance.markets = {}
        yield mock_instance

def test_symbol_matching_exact(mock_exchange):
    """Test that it uses the pre-defined symbol if it exists in markets."""
    mock_exchange.markets = {
        'BTC/USDT': {'symbol': 'BTC/USDT'},
        'BTC/USD': {'symbol': 'BTC/USD'}
    }
    
    client = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USDT')
    client._markets_loaded = True
    
    symbol = client._symbol_for_token('BTC')
    assert symbol == 'BTC/USDT'

def test_symbol_matching_fallback_preference(mock_exchange):
    """Test that it prefers USDT, then USDC, then USD if default is not available."""
    # Scenario 1: Default is USD, but only USDT exists
    mock_exchange.markets = {
        'BTC/USDT': {'symbol': 'BTC/USDT'}
    }
    client1 = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USD')
    client1._markets_loaded = True
    assert client1._symbol_for_token('BTC') == 'BTC/USDT'
    
    # Scenario 2: Default is USD, but USDT and USDC exist (prefers USDT)
    mock_exchange.markets = {
        'BTC/USDC': {'symbol': 'BTC/USDC'},
        'BTC/USDT': {'symbol': 'BTC/USDT'}
    }
    client2 = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USD')
    client2._markets_loaded = True
    assert client2._symbol_for_token('BTC') == 'BTC/USDT'

def test_load_historical_prices_cache_hit(mock_exchange, tmp_path):
    """Test that it returns cached prices without calling the exchange if they cover the range."""
    client = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USDT')
    
    # Mock file cache path to use tmp_path
    cache_file = tmp_path / "binanceus_BTC_USDT.json"
    
    # Write dummy cached prices: 10 prices spanning the last 10 minutes
    now = dt.datetime.now(dt.timezone.utc)
    cached_data = [
        [(now - dt.timedelta(minutes=10 - i)).isoformat(), 50000.0 + i]
        for i in range(10)
    ]
    
    with open(cache_file, 'w') as f:
        json.dump(cached_data, f)
        
    with patch('cryptotrading.trade.price.exchange._get_file_cache_path', return_value=cache_file):
        # We request 0.005 days (approx 7.2 minutes), which is fully covered by our 10-minute cache
        prices = client.load_historical_prices('BTC', days=0.005)
        
        # Verify no exchange calls were made
        assert not mock_exchange.fetch_ohlcv.called
        assert prices is not None
        assert len(prices) > 0
        assert np.isclose(prices[0][0], 50003.0)  # Starting around 7 minutes ago

def test_load_historical_prices_cache_miss_and_fetch(mock_exchange, tmp_path):
    """Test that it fetches from exchange and caches the results on a cache miss."""
    client = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USDT', timeframe='1m')
    cache_file = tmp_path / "binanceus_BTC_USDT.json"
    
    # Mock exchange markets and fetch_ohlcv
    mock_exchange.markets = {'BTC/USDT': {'symbol': 'BTC/USDT'}}
    client._markets_loaded = True
    
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    # Mock fetch_ohlcv to return 5 candles
    mock_exchange.fetch_ohlcv.return_value = [
        [now_ms - 5000 + i * 1000, 50000.0, 50010.0, 49990.0, 50005.0 + i, 1.0]
        for i in range(5)
    ]
    
    with patch('cryptotrading.trade.price.exchange._get_file_cache_path', return_value=cache_file):
        prices = client.load_historical_prices('BTC', days=0.001)
        
        # Verify exchange was called
        assert mock_exchange.fetch_ohlcv.called
        assert prices is not None
        assert len(prices) == 5
        assert np.isclose(prices[0][0], 50005.0)
        
        # Verify it was saved to the file cache
        assert cache_file.exists()
        with open(cache_file, 'r') as f:
            saved_data = json.load(f)
        assert len(saved_data) == 5
        assert saved_data[0][1] == 50005.0

def test_load_historical_prices_cancellation(mock_exchange, tmp_path):
    """Test that the download process can be cancelled early via cancellation_event."""
    client = ExchangePriceClient(tokens=['BTC'], exchange_id='binanceus', quote_currency='USDT', timeframe='1m')
    cache_file = tmp_path / "binanceus_BTC_USDT.json"
    
    mock_exchange.markets = {'BTC/USDT': {'symbol': 'BTC/USDT'}}
    client._markets_loaded = True
    
    # Mock fetch_ohlcv to return 1 candle
    now_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    mock_exchange.fetch_ohlcv.return_value = [
        [now_ms, 50000.0, 50000.0, 50000.0, 50000.0, 1.0]
    ]
    
    cancellation_event = threading.Event()
    
    # We will set the side_effect to set the cancellation event and return the mock value.
    def side_effect(*args, **kwargs):
        cancellation_event.set()
        return [[now_ms, 50000.0, 50000.0, 50000.0, 50000.0, 1.0]]
    mock_exchange.fetch_ohlcv.side_effect = side_effect
    
    with patch('cryptotrading.trade.price.exchange._get_file_cache_path', return_value=cache_file):
        prices = client.load_historical_prices('BTC', days=0.1, cancellation_event=cancellation_event)
        
        # Verify it terminated early
        assert mock_exchange.fetch_ohlcv.call_count == 1
        assert prices is not None
        assert len(prices) == 1
        
        # Verify the partial result was cached
        assert cache_file.exists()
