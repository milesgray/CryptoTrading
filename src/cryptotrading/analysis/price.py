import nest_asyncio
import pandas as pd
from datetime import datetime, timedelta
import pytz

from cryptotrading.data.factory import get_price_adapter
from cryptotrading.config import DB_BACKEND

class PriceAnalytics:
    def __init__(self):
        """Initialize the analytics utility with database adapter"""
        self.adapter = get_price_adapter()
        # Initialize adapter if needed
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()
        
        if loop.is_running():
            loop.create_task(self.adapter.initialize())
        else:
            loop.run_until_complete(self.adapter.initialize())
    
    def get_index_prices(self, symbol="BTC", start_time=None, end_time=None, interval="1m"):
        """
        Retrieve index prices for a specific symbol within a time range
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC")
        - start_time: Start datetime (defaults to 24 hours ago)
        - end_time: End datetime (defaults to now)
        - interval: Time interval for resampling ('1m', '5m', '15m', '1h', etc.)
        
        Returns:
        - DataFrame with timestamp and price columns
        """
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now(pytz.UTC)
        if start_time is None:
            start_time = end_time - timedelta(days=1)
            
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            
        # Get raw price data from database adapter
        if loop.is_running():
            # In an active event loop, we run in an executor or helper if async call is needed
            # For simplicity in this analytical class, we use run_until_complete if we can, or a future
            nest_asyncio.apply()
        
        data = loop.run_until_complete(self.adapter.get_price_data(symbol, start_time, end_time, limit=50000))
        
        if not data:
            return pd.DataFrame()
            
        # Standardize representation between Mongo and Postgres
        processed = []
        for doc in data:
            # Postgres returns standard dict with price/timestamp, Mongo might have it differently nested
            price = doc.get("price")
            ts = doc.get("timestamp")
            processed.append({
                "timestamp": ts,
                "price": price
            })
            
        df = pd.DataFrame(processed)
        if df.empty:
            return df
            
        # Convert to time series and resample if needed
        df.set_index("timestamp", inplace=True)
        df.sort_index(inplace=True)
        
        # Resample data based on interval
        if interval != "raw":
            # Map common interval strings to pandas offset strings
            interval_map = {
                "1m": "1Min",
                "5m": "5Min",
                "15m": "15Min",
                "30m": "30Min",
                "1h": "1H",
                "4h": "4H",
                "1d": "1D"
            }
            pandas_interval = interval_map.get(interval, interval)
            
            # Resample and forward fill
            df = df.resample(pandas_interval).last().ffill()
        
        return df
    
    def get_exchange_prices(self, symbol="BTC", exchange=None, start_time=None, end_time=None):
        """
        Retrieve prices from specific exchanges for comparison
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC")
        - exchange: Exchange ID (or None for all exchanges)
        - start_time: Start datetime
        - end_time: End datetime
        
        Returns:
        - DataFrame with timestamp, exchange, and price columns
        """
        # Set default time range if not provided
        if end_time is None:
            end_time = datetime.now(pytz.UTC)
        if start_time is None:
            start_time = end_time - timedelta(days=1)
            
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            
        if loop.is_running():
            
            nest_asyncio.apply()
            
        if DB_BACKEND == 'mongodb':
            # Mongo query directly on the raw collection since there isn't a dedicated get_exchange_prices in PriceMongoAdapter
            cursor = self.adapter.price_collection.find({
                "metadata.symbol": symbol,
                "metadata.type": "exchange_data",
                "timestamp": {
                    "$gte": start_time,
                    "$lte": end_time
                }
            }).sort("timestamp", 1)
            data = loop.run_until_complete(cursor.to_list(length=None))
            
            processed_data = []
            for item in data:
                meta = item.get("metadata", {})
                if exchange and meta.get("exchange") != exchange:
                    continue
                processed_item = {
                    "timestamp": item["timestamp"],
                    "exchange": meta.get("exchange"),
                    "price": item["price"]
                }
                processed_data.append(processed_item)
            return pd.DataFrame(processed_data)
        else:
            # Postgres: fetch raw exchange data
            query = '''
                SELECT time as timestamp, close as price, symbol, exchange, metadata
                FROM price_data
                WHERE (metadata->>'token' = $1 OR symbol LIKE $2 OR symbol = $1)
                  AND exchange LIKE 'exchange_raw_%'
                  AND time >= $3 AND time <= $4
                ORDER BY time ASC;
            '''
            from cryptotrading.data.postgres import get_connection
            
            async def run_query():
                async with get_connection() as conn:
                    return await conn.fetch(query, symbol, f"{symbol}/%", start_time, end_time)
                    
            rows = loop.run_until_complete(run_query())
            processed_data = []
            for r in rows:
                exch = r["exchange"].replace("exchange_raw_", "")
                if exchange and exch != exchange:
                    continue
                processed_data.append({
                    "timestamp": r["timestamp"],
                    "exchange": exch,
                    "price": r["price"]
                })
            return pd.DataFrame(processed_data)
        
    def calculate_vwap(self, symbol, window_hours=24):
        """
        Calculate Volume-Weighted Average Price
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
        - window_hours: Hours to include in calculation
        
        Returns:
        - VWAP value
        """
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(hours=window_hours)
        
        # Get exchange data which includes individual trades
        exchange_data = self.get_exchange_prices(symbol, None, start_time, end_time)
        
        if exchange_data.empty:
            return None
            
        # Group by timestamp and calculate weighted average
        exchange_data["volume"] = 1  # If real volume is not available, assume equal weight
        
        # Calculate VWAP
        total_volume = exchange_data["volume"].sum()
        if total_volume == 0:
            return None
            
        vwap = (exchange_data["price"] * exchange_data["volume"]).sum() / total_volume
        
        return vwap
    
    def calculate_market_impact(self, symbol, bet_amount, bet_multiplier, current_price=None):
        """
        Calculate the market impact for a given bet size using Rollbit's formula
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
        - bet_amount: Amount of the bet in USD
        - bet_multiplier: Leverage multiplier
        - current_price: Current price (if None, latest price will be used)
        
        Returns:
        - Dictionary with market impact information
        """
        # Get current price if not provided
        if current_price is None:
            latest_data = self.get_index_prices(symbol, interval="1m")
            if latest_data.empty:
                return None
            current_price = latest_data["price"].iloc[-1]
        
        # Market impact parameters (these would normally be retrieved from the API)
        base_rate = 0.85  # Example value
        rate_multiplier = 0.2  # Example value
        rate_exponent = 2.0  # Example value
        position_multiplier = 0.5  # Example value
        
        # Calculate effective position size
        position_size = bet_amount * bet_multiplier
        
        # Calculate impact for different price movements
        results = {}
        
        for pct_change in [-0.10, -0.05, -0.02, -0.01, 0.01, 0.02, 0.05, 0.10]:
            future_price = current_price * (1 + pct_change)
            
            # Calculate price ratio
            price_ratio = future_price / current_price
            price_change = price_ratio - 1
            
            # Avoid division by zero
            if price_change == 0:
                results[f"{pct_change*100:.1f}%"] = future_price
                continue
                
            # Apply market impact formula for winning trades
            if (pct_change > 0 and bet_multiplier > 0) or (pct_change < 0 and bet_multiplier < 0):
                impact_factor = (1 - base_rate) / (
                    1 + 1/abs(price_change * rate_multiplier)**rate_exponent + 
                    position_size/(10**6 * abs(price_change) * position_multiplier)
                )
                
                # Adjust closing price
                adjusted_price = current_price + impact_factor * (future_price - current_price)
                results[f"{pct_change*100:.1f}%"] = adjusted_price
            else:
                # No impact for losing trades
                results[f"{pct_change*100:.1f}%"] = future_price
        
        return {
            "current_price": current_price,
            "bet_amount": bet_amount,
            "bet_multiplier": bet_multiplier,
            "position_size": position_size,
            "market_impact": results
        }
    
    def close(self):
        """Close database adapter connection"""
        import asyncio
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop_policy().get_event_loop()
            
        if loop.is_running():
            nest_asyncio.apply()
        loop.run_until_complete(self.adapter.shutdown())
