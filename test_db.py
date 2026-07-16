import asyncio
import json
import sys
import os
import time
import datetime as dt
import dotenv

# Load environment variables before importing package
dotenv.load_dotenv(dotenv.find_dotenv())

sys.path.insert(0, os.path.abspath('src'))
import cryptotrading
from cryptotrading.data.book import OrderBookPostgresAdapter
from cryptotrading.data.price import PricePostgresAdapter

async def main():
    # 1. Test Order Book Adapter
    book_adapter = OrderBookPostgresAdapter()
    await book_adapter.initialize()
    
    token = "ETH"
    end_time = dt.datetime.now(dt.timezone.utc)
    start_time = end_time - dt.timedelta(hours=2) # 2 hours of data
    
    print(f"--- Running Order Book Performance Comparisons for token '{token}' ---")
    print(f"Time Range: {start_time} to {end_time}")
    
    t0 = time.time()
    snapshots_normal = await book_adapter.get_orderbook_data(token, start_time, end_time)
    t_normal = time.time() - t0
    print(f"Raw query (Exact Match Backward Scan):")
    print(f"  Snapshots returned: {len(snapshots_normal)}")
    print(f"  Time taken: {t_normal:.3f} seconds")
    
    t0 = time.time()
    snapshots_10s = await book_adapter.get_orderbook_data(token, start_time, end_time, interval=10.0)
    t_10s = time.time() - t0
    print(f"Downsampled query (10s time_bucket & last aggregate):")
    print(f"  Snapshots returned: {len(snapshots_10s)}")
    print(f"  Time taken: {t_10s:.3f} seconds")
    print(f"  Speedup vs raw: {t_normal / max(t_10s, 0.001):.1f}x")

    # 2. Test Price Adapter
    print("\n--- Testing Price Adapter Optimizations ---")
    price_adapter = PricePostgresAdapter()
    await price_adapter.initialize()
    
    # Test get_price_data
    t0 = time.time()
    prices = await price_adapter.get_price_data(token + "/USDT", start_time, end_time, limit=50)
    print(f"get_price_data (symbol={token}/USDT, limit=50):")
    print(f"  Prices returned: {len(prices)}")
    print(f"  Time taken: {time.time() - t0:.3f} seconds")
    if prices:
        print(f"  First price: {prices[0]['price']} at {prices[0]['timestamp']}")
        
    # Test get_price_data_count
    t0 = time.time()
    count = await price_adapter.get_price_data_count(token + "/USDT", start_time, end_time)
    print(f"get_price_data_count:")
    print(f"  Total records count: {count}")
    print(f"  Time taken: {time.time() - t0:.3f} seconds")
    
    # Test get_latest_price
    t0 = time.time()
    latest = await price_adapter.get_latest_price(token)
    print(f"get_latest_price:")
    print(f"  Latest price: {latest['price'] if latest else None}")
    print(f"  Time taken: {time.time() - t0:.3f} seconds")
    
    # Test get_candlestick_data
    t0 = time.time()
    # 60s granularity candlestick data
    candles = await price_adapter.get_candlestick_data(token, start_time, end_time, granularity=60)
    print(f"get_candlestick_data (granularity=60s):")
    print(f"  Candlesticks returned: {len(candles)}")
    print(f"  Time taken: {time.time() - t0:.3f} seconds")
    if candles:
        print(f"  First candle: O={candles[0].open}, H={candles[0].high}, L={candles[0].low}, C={candles[0].close}, V={candles[0].volume}")

if __name__ == "__main__":
    asyncio.run(main())
