import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from datetime import datetime, timedelta
import pytz

class PriceAnalytics:
    def __init__(self, mongo_uri="mongodb://localhost:27017", db_name="crypto_prices", collection_name="price_data"):
        """Initialize the analytics utility with MongoDB connection details"""
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
    
    def get_index_prices(self, symbol, start_time=None, end_time=None, interval="1m"):
        """
        Retrieve index prices for a specific symbol within a time range
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
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
            
        # Query MongoDB for index prices
        query = {
            "metadata.symbol": symbol,
            "metadata.type": "index_price",
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        
        projection = {
            "_id": 0,
            "timestamp": 1,
            "price": 1
        }
        
        # Retrieve data and convert to DataFrame
        cursor = self.collection.find(query, projection).sort("timestamp", 1)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
            
        df = pd.DataFrame(data)
        
        # Convert to time series and resample if needed
        df.set_index("timestamp", inplace=True)
        
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
    
    def get_exchange_prices(self, symbol, exchange=None, start_time=None, end_time=None):
        """
        Retrieve prices from specific exchanges for comparison
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
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
            
        # Build query
        query = {
            "metadata.symbol": symbol,
            "metadata.type": "exchange_data",
            "timestamp": {
                "$gte": start_time,
                "$lte": end_time
            }
        }
        
        if exchange:
            query["metadata.exchange"] = exchange
            
        projection = {
            "_id": 0,
            "timestamp": 1,
            "metadata.exchange": 1,
            "price": 1,
            "bid": 1,
            "ask": 1
        }
        
        # Retrieve data and convert to DataFrame
        cursor = self.collection.find(query, projection).sort("timestamp", 1)
        data = list(cursor)
        
        if not data:
            return pd.DataFrame()
            
        # Process data to flatten the metadata
        processed_data = []
        for item in data:
            processed_item = {
                "timestamp": item["timestamp"],
                "exchange": item["metadata"]["exchange"],
                "price": item["price"]
            }
            
            # Add bid and ask if available
            if "bid" in item:
                processed_item["bid"] = item["bid"]
            if "ask" in item:
                processed_item["ask"] = item["ask"]
                
            processed_data.append(processed_item)
            
        return pd.DataFrame(processed_data)
        
    def calculate_funding_rate(self, symbol, lookback_hours=24):
        """
        Calculate the funding rate based on price movements
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
        - lookback_hours: Hours to look back for calculation
        
        Returns:
        - Current hourly funding rate (as a percentage)
        """
        end_time = datetime.now(pytz.UTC)
        start_time = end_time - timedelta(hours=lookback_hours)
        
        # Get price data
        df = self.get_index_prices(symbol, start_time, end_time, interval="1h")
        
        if df.empty or len(df) < 2:
            return 0.0
            
        # Calculate price movement
        price_change_pct = (df["price"].iloc[-1] / df["price"].iloc[0] - 1) * 100
        
        # Apply a simple model for funding rate calculation
        # This is a simplified example - real funding rates are more complex
        hourly_funding_rate = price_change_pct / (4 * lookback_hours)  # Dampen the effect
        
        # Cap the funding rate
        capped_rate = np.clip(hourly_funding_rate, -0.375, 0.375)  # Cap at +/-0.375% per hour
        
        return capped_rate
    
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
    
    def plot_price_chart(self, symbol, start_time=None, end_time=None, interval="15m", include_exchanges=False):
        """
        Create a price chart for visualization
        
        Parameters:
        - symbol: Trading symbol (e.g., "BTC/USDT")
        - start_time: Start datetime
        - end_time: End datetime
        - interval: Time interval for display
        - include_exchanges: Whether to include individual exchange prices
        
        Returns:
        - Matplotlib figure object
        """
        # Get index price data
        index_data = self.get_index_prices(symbol, start_time, end_time, interval)
        
        if index_data.empty:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot index price
        ax.plot(index_data.index, index_data["price"], label="Rollbit Index Price", linewidth=2)
        
        # Plot exchange prices if requested
        if include_exchanges:
            exchange_data = self.get_exchange_prices(symbol, None, start_time, end_time)
            
            if not exchange_data.empty:
                # Group by exchange
                for exchange, group in exchange_data.groupby("exchange"):
                    # Convert to DataFrame with datetime index
                    exc_df = pd.DataFrame({"price": group["price"].values}, index=group["timestamp"])
                    
                    # Resample to match index data interval
                    exc_df = exc_df.resample(interval).mean()
                    
                    # Plot with lower alpha
                    ax.plot(exc_df.index, exc_df["price"], label=exchange, alpha=0.5, linewidth=1)
        
        # Calculate and show VWAP
        vwap = self.calculate_vwap(symbol)
        if vwap is not None:
            ax.axhline(y=vwap, color='r', linestyle='--', label=f'VWAP: {vwap:.2f}')
            
        # Calculate and show funding rate
        funding_rate = self.calculate_funding_rate(symbol)
        title = f"{symbol} Price Chart"
        title += f" (Funding Rate: {funding_rate:.4f}% per hour)" if funding_rate != 0 else ""
        
        # Style the chart
        ax.set_title(title)
        ax.set_xlabel("Time")
        ax.set_ylabel("Price (USDT)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
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
        """Close MongoDB connection"""
        if self.client:
            self.client.close()

# Example usage
if __name__ == "__main__":
    analytics = PriceAnalytics()
    
    # Get recent BTC/USDT prices
    btc_prices = analytics.get_index_prices("BTC/USDT", interval="5m")
    print(f"Recent BTC/USDT prices:\n{btc_prices.tail()}")
    
    # Calculate VWAP
    vwap = analytics.calculate_vwap("BTC/USDT")
    print(f"BTC/USDT 24h VWAP: {vwap:.2f}" if vwap else "VWAP calculation failed")
    
    # Calculate funding rate
    funding_rate = analytics.calculate_funding_rate("BTC/USDT")
    print(f"Current hourly funding rate: {funding_rate:.6f}%")
    
    # Calculate market impact for a 10,000 USD bet with 10x leverage
    impact = analytics.calculate_market_impact("BTC/USDT", 10000, 10)
    if impact:
        print("\nMarket Impact Analysis:")
        print(f"Current Price: ${impact['current_price']:.2f}")
        print(f"Position Size: ${impact['position_size']:.2f}")
        print("Expected closing prices at different market movements:")
        for movement, price in impact['market_impact'].items():
            print(f"  Market moves {movement}: ${price:.2f}")
    
    # Plot price chart
    fig = analytics.plot_price_chart("BTC/USDT", interval="15m")
    if fig:
        plt.show()
    
    # Close connection
    analytics.close()