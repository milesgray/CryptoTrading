"""
Retrieval Service FastAPI Microservice.

This module sets up a FastAPI web service that bootstraps historical 1-minute
candlestick data from CCXT exchanges into Postgres, indexes historical price
segments using neural embeddings, and serves shape-similarity forecasting queries.
"""

import datetime
import logging
import asyncio
import numpy as np
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from encoder import RetrievalServiceEncoder
from forecaster import SpecReTFForecaster
from cryptotrading.data.factory import get_price_adapter
from cryptotrading.config import SYMBOLS

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval_service")

app = FastAPI(
    title="Retrieval Service API",
    description="Microservice for real-time shape-similarity quantitative timeseries matching.",
    version="1.0.0"
)

# Initialize encoder and forecaster with dim=184 for combined embedding compatibility (128D deep learning + 56D local features)
encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=184)
forecaster = SpecReTFForecaster(encoder_service, frame_size=16, hop_size=4)

async def bootstrap_historical_data(price_adapter, symbol: str, days: int = 7):
    """
    Fetch historical 1m candlestick data from a CCXT exchange and store it in Postgres.

    Checks the database to verify if sufficient data exists for the given symbol.
    If not, it paginates backward using CCXT (trying Binance, Coinbase, or Kraken)
    to pull 7 days of 1-minute candlesticks. The candles are written to the database
    with exchange set to 'index' using ON CONFLICT DO NOTHING to prevent duplicates.

    Args:
        price_adapter: DB adapter instance used to query/insert prices.
        symbol (str): Target trading symbol (e.g. 'BTC/USDT').
        days (int): Number of historical days to pull. Defaults to 7.

    Raises:
        Exception: Propagates CCXT or database connection failures.
    """
    import ccxt
    from cryptotrading.data.postgres import get_connection

    now = datetime.datetime.now(datetime.timezone.utc)
    start_time = now - datetime.timedelta(days=days)

    # Check database for existing data points in timeframe
    count = await price_adapter.get_price_data_count(symbol, start_time, now)
    logger.info(f"Existing price data points in database for {symbol}: {count}")

    expected_points = days * 24 * 60
    if count >= expected_points * 0.8:
        logger.info(f"Sufficient data already exists for {symbol} ({count}/{expected_points} points). Skipping CCXT fetch.")
        return

    logger.info(f"Bootstrapping historical 1m candlestick data for {symbol} from CCXT...")

    # Identify and load the exchange
    exchanges_to_try = ["binance", "coinbase", "kraken"]
    exchange = None
    cex = None
    for ex_name in exchanges_to_try:
        try:
            cex = getattr(ccxt, ex_name)()
            cex.load_markets()
            if symbol in cex.markets:
                exchange = ex_name
                break
        except Exception as e:
            logger.warning(f"Failed to check exchange {ex_name}: {e}")

    if not cex:
        cex = ccxt.binance()
        exchange = "binance"

    logger.info(f"Using exchange {exchange} to pull historical data for {symbol}")

    limit = 1000
    since = int((now - datetime.timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    
    max_retries = 3
    retries = 0
    current_since = since
    target_timestamp_ms = int(now.timestamp() * 1000)

    while current_since < target_timestamp_ms:
        try:
            logger.info(f"Fetching candles starting from {datetime.datetime.fromtimestamp(current_since/1000, tz=datetime.timezone.utc)}")
            ohlcv = cex.fetch_ohlcv(symbol, '1m', since=current_since, limit=limit)
            if not ohlcv:
                logger.info("No more candles returned by CEX.")
                break

            all_candles.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            if last_timestamp <= current_since:
                current_since += 60 * 1000
            else:
                current_since = last_timestamp + 60 * 1000

            await asyncio.sleep(cex.rateLimit / 1000.0 if hasattr(cex, 'rateLimit') else 1.0)
            retries = 0
        except Exception as e:
            logger.error(f"Error fetching OHLCV from {exchange}: {e}")
            retries += 1
            if retries >= max_retries:
                logger.error("Max retries reached while fetching OHLCV. Stopping.")
                break
            await asyncio.sleep(2 * retries)

    if not all_candles:
        logger.warning(f"No candles fetched from CCXT for {symbol}.")
        return

    logger.info(f"Fetched {len(all_candles)} candles from CCXT. Inserting into Postgres...")

    inserted_count = 0
    async with get_connection() as conn:
        async with conn.transaction():
            for candle in all_candles:
                ts_ms, open_p, high_p, low_p, close_p, volume = candle
                dt_val = datetime.datetime.fromtimestamp(ts_ms / 1000, tz=datetime.timezone.utc)
                token = symbol.split("/")[0] if "/" in symbol else symbol
                
                metadata = {
                    "token": token,
                    "symbol": symbol,
                    "type": "index_price",
                    "price": close_p,
                    "exchanges_count": 1,
                    "bootstrapped": True
                }

                await conn.execute('''
                    INSERT INTO price_data (time, symbol, exchange, open, high, low, close, volume, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                    ON CONFLICT (time, symbol, exchange) DO NOTHING;
                ''', dt_val, symbol, 'index', open_p, high_p, low_p, close_p, volume, metadata)
                inserted_count += 1

    logger.info(f"Successfully processed {inserted_count} candles for {symbol} (with ON CONFLICT DO NOTHING).")

# Dynamic cache for forecasters by (token, granularity_seconds, window_size)
forecasters_cache = {}
cache_lock = asyncio.Lock()

async def build_index_for_combination(token: str, granularity_sec: int, window_size: int) -> SpecReTFForecaster:
    """
    Build a retrieval index for the given combination of token, granularity, and window size.
    """
    logger.info(f"Building dynamic retrieval index for token={token}, granularity={granularity_sec}s, window_size={window_size}...")
    price_adapter = get_price_adapter()
    await price_adapter.initialize()
    
    end_time = datetime.datetime.now(datetime.timezone.utc)
    # Determine history days to load: at least 7 days, or more if granularity is high
    min_duration_sec = max(7 * 24 * 3600, window_size * 4 * granularity_sec)
    start_time = end_time - datetime.timedelta(seconds=min_duration_sec)
    
    candles = await price_adapter.get_candlestick_data(
        token=token,
        start_time=start_time,
        end_time=end_time,
        granularity=granularity_sec,
        include_book=True
    )
    
    if not candles or len(candles) <= window_size:
        raise ValueError(
            f"Insufficient historical data to build index for token {token} at granularity {granularity_sec}s. "
            f"Found {len(candles) if candles else 0} points, need at least {window_size + 5}."
        )
        
    # Determine parameters dynamically
    horizon = window_size
    if len(candles) < (window_size + horizon):
        horizon = max(5, len(candles) - window_size)
        
    n_fft = 32
    while n_fft >= window_size:
        n_fft = n_fft // 2
    n_fft = max(8, n_fft)
    
    frame_size = 16
    while frame_size >= window_size:
        frame_size = frame_size // 2
    frame_size = max(4, frame_size)
    hop_size = max(1, frame_size // 4)
    
    logger.info(f"Initializing encoder with window_size={window_size}, n_fft={n_fft}, frame_size={frame_size}, hop_size={hop_size}, horizon={horizon}")
    encoder = RetrievalServiceEncoder(window_size=window_size, n_fft=n_fft, dim=184)
    
    # Build sliding window segments in batch
    prices_list = []
    order_books = []
    metadatas = []
    
    for i in range(len(candles) - window_size - horizon + 1):
        window = candles[i : i + window_size]
        future_window = candles[i + window_size : i + window_size + horizon]
        
        prices = np.array([float(c.close) for c in window])
        future_prices = np.array([float(c.close) for c in future_window])
        
        last_candle = window[-1]
        order_book = {}
        if last_candle.order_book:
            ob = last_candle.order_book
            order_book = {
                "bids": [[b.avg_price, b.volume] for b in ob.bid_buckets] if hasattr(ob, 'bid_buckets') else [],
                "asks": [[a.avg_price, a.volume] for a in ob.ask_buckets] if hasattr(ob, 'ask_buckets') else []
            }
        
        if not order_book or not order_book.get("bids"):
            order_book = {"bids": [[prices[-1], 1.0]], "asks": [[prices[-1] + 1.0, 1.0]]}
            
        prices_list.append(prices)
        order_books.append(order_book)
        metadatas.append({
            "id": i,
            "historical_prices": prices.tolist(),
            "prices": future_prices.tolist(),
            "order_book": order_book
        })
        
    if prices_list:
        encoder.add_segments_batch(prices_list, order_books, metadatas)
        
    encoder.build_index(n_trees=10)
    
    new_forecaster = SpecReTFForecaster(
        encoder_service=encoder,
        frame_size=frame_size,
        hop_size=hop_size,
        horizon=horizon
    )
    return new_forecaster

async def get_forecaster(token: str, granularity_sec: int, window_size: int) -> SpecReTFForecaster:
    key = (token, granularity_sec, window_size)
    async with cache_lock:
        if key not in forecasters_cache:
            forecaster_instance = await build_index_for_combination(token, granularity_sec, window_size)
            forecasters_cache[key] = forecaster_instance
        return forecasters_cache[key]

@app.on_event("startup")
async def startup_event():
    """
    FastAPI Startup Event Handler.

    Initializes the database adapters, triggers CCXT historical data bootstrapping,
    queries the database to retrieve historical prices, parses sliding window segments,
    indexes them using the encoder service, and constructs the vector index.
    
    If bootstrapping fails or database is empty, the service aborts startup with an
    exception (no mock data fallback is permitted).
    """
    logger.info("Initializing database adapters...")
    price_adapter = get_price_adapter()
    await price_adapter.initialize()
    
    # Bootstrap historical data via CCXT if missing
    for symbol in SYMBOLS:
        try:
            await bootstrap_historical_data(price_adapter, symbol, days=7)
        except Exception as e:
            logger.error(f"Failed to bootstrap historical data for {symbol}: {e}", exc_info=True)
    
    # Build default index for each configured symbol at (60, 60)
    global encoder_service, forecaster
    default_forecaster = None
    for symbol in SYMBOLS:
        token = symbol.split("/")[0] if "/" in symbol else symbol
        try:
            logger.info(f"Pre-building default retrieval index for {token}...")
            f_inst = await get_forecaster(token, granularity_sec=60, window_size=60)
            if token == "BTC":
                default_forecaster = f_inst
        except Exception as e:
            logger.error(f"Failed to build startup index for {token}: {e}", exc_info=True)
            raise e
            
    if default_forecaster:
        forecaster = default_forecaster
        encoder_service = default_forecaster.encoder_service
    else:
        logger.warning("BTC default forecaster not pre-built, globals left as is.")

@app.get("/forecast")
async def forecast(
    symbol: str = "BTC",
    k: int = 5,
    granularity: str = "1m",
    window_size: int = 60
) -> Dict[str, Any]:
    """
    Get matching historical patterns and compute a consensus forecast.

    Retrieves the last window_size price points of live price data for the specified token
    at the specified granularity, queries the dynamic vector index for similar historical cycles,
    and returns a similarity-weighted projection.

    Args:
        symbol (str): The token/symbol to query. Defaults to 'BTC'.
        k (int): Number of matches to retrieve. Defaults to 5.
        granularity (str): The candlestick interval (e.g. '1m', '5m', '15m', '1h'). Defaults to '1m'.
        window_size (int): Temporal window width of sequence. Defaults to 60.

    Returns:
        Dict[str, Any]: Forecast predictions, consensus path, and similarity stats.

    Raises:
        HTTPException (400): If there is insufficient live query data.
        HTTPException (500): For general retrieval and forecasting processing failures.
    """
    try:
        price_adapter = get_price_adapter()
        await price_adapter.initialize()
        
        token = symbol.split("/")[0] if "/" in symbol else symbol
        
        # Map granularity string/integer to seconds
        gran_map = {
            "1m": 60,
            "5m": 300,
            "15m": 900,
            "1h": 3600
        }
        
        if isinstance(granularity, str):
            try:
                granularity_sec = int(granularity)
            except ValueError:
                granularity_sec = gran_map.get(granularity.lower(), 60)
        else:
            granularity_sec = int(granularity)
            
        # Calculate time window to fetch live prices
        end_time = datetime.datetime.now(datetime.timezone.utc)
        duration_sec = window_size * granularity_sec * 2
        start_time = end_time - datetime.timedelta(seconds=duration_sec)
        
        candles = await price_adapter.get_candlestick_data(
            token=token,
            start_time=start_time,
            end_time=end_time,
            granularity=granularity_sec,
            include_book=True
        )
        
        if not candles or len(candles) < window_size:
            raise HTTPException(
                status_code=400,
                detail=f"Insufficient live query data for {token} at {granularity} granularity. "
                       f"Need at least {window_size} points, found {len(candles) if candles else 0}."
            )
            
        query_candles = candles[-window_size:]
        current_prices = np.array([float(c.close) for c in query_candles])
        
        last_candle = query_candles[-1]
        current_order_book = {}
        if last_candle.order_book:
            ob = last_candle.order_book
            current_order_book = {
                "bids": [[b.avg_price, b.volume] for b in ob.bid_buckets] if hasattr(ob, 'bid_buckets') else [],
                "asks": [[a.avg_price, a.volume] for a in ob.ask_buckets] if hasattr(ob, 'ask_buckets') else []
            }
        if not current_order_book or not current_order_book.get("bids"):
            current_order_book = {"bids": [[current_prices[-1], 1.0]], "asks": [[current_prices[-1] + 1.0, 1.0]]}
        
        dyn_forecaster = await get_forecaster(token, granularity_sec, window_size)
        return dyn_forecaster.forecast(current_prices, current_order_book, k=k)
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)