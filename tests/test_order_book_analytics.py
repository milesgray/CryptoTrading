import pytest
import math
import numpy as np
import pandas as pd
from unittest.mock import MagicMock

from cryptotrading.rollbit.prices.formula import calculate_index_price
from cryptotrading.rollbit.prices.metrics import calculate_multi_exchange_metrics
from cryptotrading.rollbit.prices.book import OrderBookManager
from cryptotrading.util.status import StatusManager

@pytest.fixture
def sample_valid_order_books():
    # 6 valid, identical order books to meet the min_valid_feeds requirement of 6
    book = {
        'bids': [(100.0, 1000.0), (99.0, 500.0)],
        'asks': [(101.0, 1000.0), (102.0, 500.0)],
        'exchange': 'test_exchange'
    }
    return [dict(book) for _ in range(6)]

def test_calculate_index_price_success(sample_valid_order_books):
    price_info = calculate_index_price(sample_valid_order_books, min_valid_feeds=6, return_book=True)
    assert price_info is not None
    assert 'price' in price_info
    assert 'size' in price_info
    assert 'book' in price_info
    
    # Mid-price should be close to 100.5
    assert 100.0 < price_info['price'] < 101.0
    # Total composite size should be capped size sum: (1000 + 500 + 1000 + 500) * 6 = 18000
    assert price_info['size'] == 18000.0

def test_calculate_index_price_outlier_filtering(sample_valid_order_books):
    # Add an outlier exchange book with a mid price deviating by > 10%
    outlier_book = {
        'bids': [(120.0, 1000.0), (119.0, 500.0)],  # Mid price = 120.5 (deviates by 20% from 100.5)
        'asks': [(121.0, 1000.0), (122.0, 500.0)],
        'exchange': 'outlier_exchange'
    }
    books_with_outlier = sample_valid_order_books + [outlier_book]
    
    # First, run calculation with min_valid_feeds=6
    res = calculate_index_price(books_with_outlier, min_valid_feeds=6, return_book=True)
    assert res is not None
    
    # The composite book should not contain outlier bids/asks in the composite
    # Check that bids in the composite book don't contain prices close to 120
    bids = res['book']['bids']
    prices = [b[0] for b in bids]
    assert all(p < 110.0 for p in prices)

def test_calculate_index_price_crossed_filtering(sample_valid_order_books):
    # Add a crossed book (best_bid > best_ask)
    crossed_book = {
        'bids': [(102.0, 1000.0)],
        'asks': [(101.0, 1000.0)],
        'exchange': 'crossed_exchange'
    }
    books_with_crossed = sample_valid_order_books + [crossed_book]
    res = calculate_index_price(books_with_crossed, min_valid_feeds=6, return_book=True)
    assert res is not None
    
    # The crossed book should be discarded and not part of the composite book
    # Total bids size should still match 6 valid books: 1500 * 6 = 9000
    bids_sum = sum(b[1] for b in res['book']['bids'])
    assert bids_sum == 9000.0

def test_calculate_index_price_min_feeds(sample_valid_order_books):
    # Only 5 valid books
    short_books = sample_valid_order_books[:5]
    res = calculate_index_price(short_books, min_valid_feeds=6)
    assert res is None

def test_calculate_index_price_invalid_inputs():
    assert calculate_index_price("not a list") is None
    assert calculate_index_price([{"bids": "not a list", "asks": []}]) is None
    assert calculate_index_price([{"bids": [(-100.0, 1000.0)], "asks": [(100.0, 1000.0)]}]) is None

def test_multi_exchange_metrics():
    order_books = [
        {
            'exchange': 'binance',
            'bids': [(100.0, 1000.0), (10.0, 5000.0)],  # 10.0 is way out of the money (> 1% threshold)
            'asks': [(101.0, 1000.0)]
        },
        {
            'exchange': 'coinbase',
            'bids': [(99.8, 2000.0)],
            'asks': [(100.8, 2000.0)]
        },
        {
            'exchange': 'kraken',
            'bids': [(100.2, 1500.0)],
            'asks': [(101.2, 1500.0)]
        }
    ]
    
    metrics = calculate_multi_exchange_metrics(order_books, index_price=100.5)
    
    # Mid prices: binance=100.5, coinbase=100.3, kraken=100.7
    # Mean mid: 100.5
    # Variance: ((100.5-100.5)^2 + (100.3-100.5)^2 + (100.7-100.5)^2) / 3 = (0 + 0.04 + 0.04) / 3 = 0.08 / 3
    # Price dispersion: sqrt(0.08 / 3) ≈ 0.1633
    assert math.isclose(metrics['price_dispersion'], math.sqrt(0.08 / 3.0), rel_tol=1e-5)
    
    # Average spread: (1.0 + 1.0 + 1.0) / 3 = 1.0
    assert metrics['average_bid_ask_spread'] == 1.0
    
    # Check HHI (depths: binance=2000 [5000 is ignored], coinbase=4000, kraken=3000 -> total=9000)
    # Shares: binance=22.22%, coinbase=44.44%, kraken=33.33%
    # HHI: 22.22^2 + 44.44^2 + 33.33^2 ≈ 493.8 + 1975.3 + 1111.1 = 3580.2
    assert 3500.0 < metrics['liquidity_concentration_hhi'] < 3600.0

def test_multi_exchange_arbitrage():
    order_books = [
        {
            'exchange': 'exchange_A',
            'bids': [(102.0, 1000.0)],  # Bid is higher than exchange_B ask!
            'asks': [(103.0, 1000.0)]
        },
        {
            'exchange': 'exchange_B',
            'bids': [(98.0, 1000.0)],
            'asks': [(99.0, 1000.0)]  # Ask is 99.0
        }
    ]
    
    metrics = calculate_multi_exchange_metrics(order_books, fee_rate=0.0015)
    assert math.isclose(metrics['max_arbitrage_spread'], 2.6985, rel_tol=1e-5)
    assert len(metrics['arbitrage_opportunities']) == 1
    opp = metrics['arbitrage_opportunities'][0]
    assert opp['buy_exchange'] == 'exchange_B'
    assert opp['sell_exchange'] == 'exchange_A'
    assert math.isclose(opp['raw_spread'], 3.0, rel_tol=1e-5)
    assert math.isclose(opp['spread'], 2.6985, rel_tol=1e-5)
    assert math.isclose(opp['fee_cost'], 0.3015, rel_tol=1e-5)
    assert opp['buy_price'] == 99.0
    assert opp['sell_price'] == 102.0

def test_validate_feeds_correction():
    # Test validate_feeds in OrderBookManager
    # Mocking self.data to avoid db issues
    db_mock = MagicMock()
    
    # Create OrderBookManager
    manager = OrderBookManager(
        symbol='BTC/USDT',
        exchange_ids=['binance', 'coinbase'],
        status=StatusManager('test')
    )
    manager.data = db_mock
    
    # Mock book feeds
    books = [
        {
            'exchange': 'binance',
            'timestamp': 1000.0,
            'bids': [(100.0, 10.0), (99.0, 5.0)],
            'asks': [(101.0, 10.0), (102.0, 5.0)]
        },
        {
            'exchange': 'coinbase',
            'timestamp': 1000.0,
            'bids': [(99.0, 20.0), (98.0, 10.0)],
            'asks': [(102.0, 20.0), (103.0, 10.0)]
        }
    ]
    
    # Mock time.time to prevent staleness filter (timestamp is 1000.0, so time.time() * 1000 must be close)
    import time
    orig_time = time.time
    time.time = lambda: 1.0  # 1.0s = 1000.0ms
    
    try:
        valid_books = manager.validate_feeds(books)
        assert len(valid_books) == 2
        # Verify both books are valid
        assert valid_books[0]['exchange'] == 'binance'
        assert valid_books[1]['exchange'] == 'coinbase'
    finally:
        time.time = orig_time
