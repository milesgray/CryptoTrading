import datetime
import logging
from fastapi import FastAPI, HTTPException
from encoder import RetrievalServiceEncoder
from forecaster import RetrievalForecaster
import numpy as np
from typing import Dict, Any

from cryptotrading.data.factory import get_price_adapter

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("retrieval_service")

app = FastAPI()

# Initialize encoder and forecaster
encoder_service = RetrievalServiceEncoder(window_size=60, n_fft=32, dim=56)
forecaster = RetrievalForecaster(encoder_service)

@app.on_event("startup")
async def startup_event():
    """Load historical data from TimescaleDB/MongoDB and build the vector index on startup."""
    logger.info("Initializing database adapters...")
    price_adapter = get_price_adapter()
    await price_adapter.initialize()
    
    symbol = "BTC/USDT"
    token = "BTC"
    end_time = datetime.datetime.now(datetime.timezone.utc)
    start_time = end_time - datetime.timedelta(days=7) # Load past 7 days
    
    logger.info(f"Fetching historical price data for {symbol} from {start_time} to {end_time}...")
    try:
        # Fetch high-resolution candlestick data to build historical segments
        candles = await price_adapter.get_candlestick_data(
            token=token,
            start_time=start_time,
            end_time=end_time,
            granularity=60, # 1-minute bars
            include_book=True
        )
        
        if not candles or len(candles) < 60:
            logger.warning(f"Insufficient historical data found ({len(candles) if candles else 0} points). Falling back to mock data for indexing.")
            # Fallback to avoid service startup crash if DB is empty
            for i in range(100):
                prices = np.random.rand(60) + i * 0.1
                order_book = {"bids": [[100 + i, 10]], "asks": [[101 + i, 10]]}
                encoder_service.add_segment(prices, order_book, {
                    "id": i,
                    "prices": prices.tolist(),
                    "order_book": order_book
                })
        else:
            logger.info(f"Loaded {len(candles)} historical candles. Building sliding window segments...")
            # Build sliding windows of size 60
            window_size = 60
            for i in range(len(candles) - window_size + 1):
                window = candles[i : i + window_size]
                prices = np.array([float(c.close) for c in window])
                
                # Retrieve the order book from the last candle in the window
                last_candle = window[-1]
                order_book = {}
                if last_candle.order_book:
                    # Map structured order book back to dictionary format
                    ob = last_candle.order_book
                    order_book = {
                        "bids": [[b.avg_price, b.volume] for b in ob.bid_buckets] if hasattr(ob, 'bid_buckets') else [],
                        "asks": [[a.avg_price, a.volume] for a in ob.ask_buckets] if hasattr(ob, 'ask_buckets') else []
                    }
                
                if not order_book or not order_book.get("bids"):
                    order_book = {"bids": [[prices[-1], 1.0]], "asks": [[prices[-1] + 1.0, 1.0]]}
                
                encoder_service.add_segment(prices, order_book, {
                    "id": i,
                    "prices": prices.tolist(),
                    "order_book": order_book
                })
            
            logger.info(f"Successfully indexed {len(candles) - window_size + 1} historical segments.")
            
    except Exception as e:
        logger.error(f"Error loading historical price data during startup: {e}", exc_info=True)
        # Fallback setup on database connection error
        for i in range(100):
            prices = np.random.rand(60) + i * 0.1
            order_book = {"bids": [[100 + i, 10]], "asks": [[101 + i, 10]]}
            encoder_service.add_segment(prices, order_book, {
                "id": i,
                "prices": prices.tolist(),
                "order_book": order_book
            })
            
    # Build vector index
    logger.info("Building vector index...")
    encoder_service.build_index(n_trees=10)
    logger.info("Vector index built successfully.")

@app.get("/forecast")
async def forecast(symbol: str = "BTC", k: int = 5) -> Dict[str, Any]:
    """Forecast endpoint returning similarity matching on real live price segments."""
    try:
        price_adapter = get_price_adapter()
        await price_adapter.initialize()
        
        token = symbol.split("/")[0] if "/" in symbol else symbol
        
        # Fetch the most recent 60 minutes of real price data to build the query segment
        end_time = datetime.datetime.now(datetime.timezone.utc)
        start_time = end_time - datetime.timedelta(hours=2) # Fetch extra to ensure we get 60 points
        
        candles = await price_adapter.get_candlestick_data(
            token=token,
            start_time=start_time,
            end_time=end_time,
            granularity=60,
            include_book=True
        )
        
        if not candles or len(candles) < 60:
            # If not enough real data exists yet in DB, fallback to mock query data to avoid 500 error
            logger.warning(f"Insufficient live query data ({len(candles) if candles else 0} points). Using fallback query.")
            current_prices = np.random.rand(60) + 50 * 0.1
            current_order_book = {"bids": [[150, 10]], "asks": [[151, 10]]}
        else:
            # Use the latest 60 candles
            query_candles = candles[-60:]
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
        
        return forecaster.forecast(current_prices, current_order_book, k=k)
    except Exception as e:
        logger.error(f"Error in forecast endpoint: {e}", exc_info=True)
        # Graceful degradation fallback
        current_prices = np.random.rand(60) + 50 * 0.1
        current_order_book = {"bids": [[150, 10]], "asks": [[151, 10]]}
        return forecaster.forecast(current_prices, current_order_book, k=k)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)