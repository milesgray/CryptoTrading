"""
Simulated Price Client for generating realistic cryptocurrency price data.

Generates tick-level price data using Geometric Brownian Motion with microstructure
noise, matching the PriceServerClient interface. Useful for testing and backtesting
when real exchange data is not available or needed.
"""

import asyncio
import threading
import time
import datetime as dt
import numpy as np
import logging
from typing import List, Optional, Tuple, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimulatedPriceClient:
    """
    Client to generate simulated tick-level cryptocurrency price data.
    
    Generates realistic price movements at ~1 second resolution using Geometric
    Brownian Motion with microstructure noise. Matches the PriceServerClient interface
    for drop-in compatibility with trading environments.
    
    Data Generation:
        - Uses GBM with crypto-realistic volatility parameters
        - Adds microstructure noise to simulate tick-level jitter
        - Supports both historical data generation and live streaming
        - Default 1-second tick resolution
    
    Args:
        tokens: List of token symbols (e.g., ['BTC', 'ETH']).
        base_prices: Dict of base prices for each token (e.g., {'BTC': 100000}).
        timeframe: Data resolution (default: '1s' for 1-second ticks).
        annual_volatility: Annual volatility (default: 0.80 = 80%).
        annual_drift: Annual drift/trend (default: 0.0 = neutral).
        microstructure_noise: Tick-level noise amplitude (default: 0.0001 = 0.01%).
        random_seed: Random seed for reproducibility (default: None).
    
    Example:
        >>> client = SimulatedPriceClient(
        ...     tokens=['BTC', 'ETH'],
        ...     base_prices={'BTC': 100000, 'ETH': 3500}
        ... )
        >>> prices = client.load_historical_prices('BTC', days=1, max_prices=86400)
        >>> # Returns 86400 ticks (1 day at 1s resolution)
        >>> prices.shape
        (86400, 2)  # (price, timestamp) pairs
    """
    
    # Map common timeframes to milliseconds
    TIMEFRAME_MS = {
        '1s': 1_000,
        '1m': 60_000,
        '3m': 180_000,
        '5m': 300_000,
        '15m': 900_000,
        '30m': 1_800_000,
        '1h': 3_600_000,
        '2h': 7_200_000,
        '4h': 14_400_000,
        '6h': 21_600_000,
        '8h': 28_800_000,
        '12h': 43_200_000,
        '1d': 86_400_000,
        '3d': 259_200_000,
        '1w': 604_800_000,
    }
    
    # Default base prices for common cryptocurrencies
    DEFAULT_BASE_PRICES = {
        'BTC': 100000.0,
        'ETH': 3500.0,
        'SOL': 200.0,
        'BNB': 700.0,
        'XRP': 2.5,
        'ADA': 1.0,
        'DOGE': 0.4,
        'AVAX': 40.0,
        'DOT': 8.0,
        'LINK': 25.0,
    }
    
    def __init__(
        self,
        tokens: List[str],
        base_prices: Optional[Dict[str, float]] = None,
        timeframe: str = '1s',
        annual_volatility: float = 0.80,
        annual_drift: float = 0.0,
        microstructure_noise: float = 0.0001,
        random_seed: Optional[int] = None,
    ):
        from . import PriceClientStatus
        
        self.tokens = tokens
        self.timeframe = timeframe
        self.annual_volatility = annual_volatility
        self.annual_drift = annual_drift
        self.microstructure_noise = microstructure_noise
        
        # Set random seed for reproducibility
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Initialize base prices
        self.base_prices = base_prices or {}
        for token in tokens:
            if token not in self.base_prices:
                self.base_prices[token] = self.DEFAULT_BASE_PRICES.get(token, 100.0)
        
        # Current prices for live streaming
        self.current_prices: Dict[str, Tuple[float, float]] = {
            token: (self.base_prices[token], time.time()) for token in tokens
        }
        
        # Price storage - matches PriceServerClient interface
        self._historical_cache: Dict[str, dict] = {
            token: {
                'timestamps': [],
                'prices': [],
                'min_time': None,
                'max_time': None
            } for token in tokens
        }
        self.historical_data: Dict[str, List[Tuple[float, str]]] = {
            token: [] for token in tokens
        }
        
        # Connection state for live streaming
        self.websocket = None
        self.connected = False
        self.loop = None
        self.thread = None
        self._stop_event = threading.Event()
        
        # Status tracking
        self.status = PriceClientStatus()
        self.status.logs = {"INFO": [], "WARNING": [], "ERROR": [], "DEBUG": []}
        self.status.running = True
        self.status.start_time = time.time()
        self.status.add_log('INFO', 'SimulatedPriceClient initialized')

    def _generate_tick_prices(
        self,
        token: str,
        num_ticks: int,
        start_time: dt.datetime,
        start_price: Optional[float] = None
    ) -> List[Tuple[float, str]]:
        """
        Generate simulated tick price data using Geometric Brownian Motion.
        
        Creates realistic 1-second resolution price movements with microstructure noise.
        
        Args:
            token: Token symbol
            num_ticks: Number of price ticks to generate
            start_time: Starting timestamp
            start_price: Starting price (uses base price if None)
            
        Returns:
            List of (price, timestamp_iso) tuples at tick resolution
        """
        base_price = start_price if start_price is not None else self.base_prices[token]
        timeframe_ms = self.TIMEFRAME_MS.get(self.timeframe, 1_000)
        
        # GBM parameters scaled for the timeframe
        seconds_per_year = 365 * 24 * 3600
        dt_step = (timeframe_ms / 1000) / seconds_per_year
        
        mu = self.annual_drift
        sigma = self.annual_volatility
        
        prices = [base_price]
        timestamps = [start_time]
        
        for i in range(1, num_ticks):
            # GBM step
            z = np.random.normal(0, 1)
            drift = (mu - 0.5 * sigma**2) * dt_step
            diffusion = sigma * np.sqrt(dt_step) * z
            
            # Add microstructure noise for tick-level realism
            noise = np.random.normal(0, self.microstructure_noise)
            
            new_price = prices[-1] * np.exp(drift + diffusion + noise)
            prices.append(new_price)
            timestamps.append(timestamps[-1] + dt.timedelta(milliseconds=timeframe_ms))
        
        result = [
            (float(p), ts.isoformat())
            for p, ts in zip(prices, timestamps)
        ]
        
        self.status.add_log('DEBUG',
            f'Generated {num_ticks} simulated ticks for {token} '
            f'(base: {base_price:.2f}, range: {min(prices):.2f} - {max(prices):.2f})')
        
        return result

    def _update_historical_cache(self, token: str, new_prices: List[Tuple[float, str]]):
        """
        Update the historical cache with new price data.
        
        Args:
            token: The token symbol
            new_prices: List of (price, timestamp_str) tuples
        """
        if not new_prices or token not in self._historical_cache:
            return
            
        cache = self._historical_cache[token]
        
        # Convert timestamps to datetime objects
        price_objs = []
        for price, ts_str in new_prices:
            try:
                if isinstance(ts_str, (int, float)):
                    ts = dt.datetime.fromtimestamp(ts_str, tz=dt.timezone.utc)
                elif isinstance(ts_str, str):
                    ts_str_clean = ts_str.replace('Z', '+00:00')
                    ts = dt.datetime.fromisoformat(ts_str_clean)
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=dt.timezone.utc)
                else:
                    continue
                price_objs.append((price, ts))
            except (ValueError, TypeError) as e:
                self.status.add_log('WARNING', f'Failed to parse timestamp {ts_str}: {e}')
        
        if not price_objs:
            return
            
        price_objs.sort(key=lambda x: x[1])
        
        if not cache['timestamps']:
            cache['prices'] = [p[0] for p in price_objs]
            cache['timestamps'] = [p[1] for p in price_objs]
            cache['min_time'] = price_objs[0][1]
            cache['max_time'] = price_objs[-1][1]
        else:
            all_prices = list(zip(cache['prices'], cache['timestamps']))
            all_prices.extend(price_objs)
            
            seen = {}
            for price, ts in all_prices:
                seen[ts] = price
                
            sorted_items = sorted(seen.items(), key=lambda x: x[0])
            
            cache['timestamps'] = [item[0] for item in sorted_items]
            cache['prices'] = [item[1] for item in sorted_items]
            cache['min_time'] = cache['timestamps'][0]
            cache['max_time'] = cache['timestamps'][-1]
        
        self.historical_data[token] = [
            (price, ts.isoformat()) 
            for price, ts in zip(cache['prices'], cache['timestamps'])
        ]

    def _get_cached_prices(
        self, 
        token: str, 
        start_time: dt.datetime, 
        end_time: dt.datetime
    ) -> Optional[List[Tuple[float, dt.datetime]]]:
        """Get prices from cache for the specified time range."""
        if token not in self._historical_cache:
            return None
            
        cache = self._historical_cache[token]
        
        if not cache['timestamps']:
            return None
            
        if (cache['max_time'] < start_time) or (cache['min_time'] > end_time):
            return None
            
        # Binary search for start
        left, right = 0, len(cache['timestamps'])
        while left < right:
            mid = (left + right) // 2
            if cache['timestamps'][mid] < start_time:
                left = mid + 1
            else:
                right = mid
        start_idx = left
        
        # Binary search for end
        left, right = 0, len(cache['timestamps'])
        while left < right:
            mid = (left + right) // 2
            if cache['timestamps'][mid] <= end_time:
                left = mid + 1
            else:
                right = mid
        end_idx = left
        
        if start_idx >= end_idx:
            return None
            
        result = [
            (cache['prices'][i], cache['timestamps'][i])
            for i in range(start_idx, end_idx)
        ]
        
        self.status.add_log('DEBUG', f'Cache hit for {token}: found {len(result)} prices.')
        
        return result

    async def _simulate_live_prices(self):
        """Simulate live price updates in real-time."""
        self.status.connected = True
        self.status.add_log('INFO', 'Starting live price simulation...')
        
        timeframe_seconds = self.TIMEFRAME_MS.get(self.timeframe, 1_000) / 1000
        
        while not self._stop_event.is_set() and self.status.connected:
            try:
                for token in self.tokens:
                    # Generate next tick based on last price
                    last_price, _ = self.current_prices[token]
                    current_time = dt.datetime.now(dt.timezone.utc)
                    
                    # Generate single new tick
                    new_tick = self._generate_tick_prices(
                        token, 
                        num_ticks=2,  # Generate 2 to get the step
                        start_time=current_time,
                        start_price=last_price
                    )
                    
                    if len(new_tick) >= 2:
                        price, timestamp = new_tick[1]
                        self.current_prices[token] = (price, time.time())
                        self.historical_data[token].append((price, timestamp))
                        
                        # Trim historical data to prevent memory bloat
                        if len(self.historical_data[token]) > 10000:
                            self.historical_data[token] = self.historical_data[token][-5000:]
                        
                        self.status.add_log('DEBUG',
                            f'Simulated price for {token}: {price:.4f} at {timestamp}')
                
                await asyncio.sleep(timeframe_seconds)
                
            except Exception as e:
                self.status.add_log('ERROR', f'Live simulation error: {e}')
                await asyncio.sleep(1)

    async def connect_websocket(self):
        """Start simulated WebSocket connection for live price streaming."""
        try:
            self.status.add_log('INFO', 'Connecting to simulated price stream...')
            await self._simulate_live_prices()
        except Exception as e:
            self.status.last_error = f"Simulated stream failed: {e}"
            self.status.add_log('ERROR', f"Simulated stream failed: {e}")
            self.status.connected = False

    def start_websocket(self):
        """Start the simulated WebSocket in the current thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.connect_websocket())
        except Exception as e:
            self.status.last_error = f"Error running simulation loop: {e}"
            self.status.add_log('ERROR', f"Error running simulation loop: {e}")
        finally:
            self.loop.close()

    def start_background_websocket(self):
        """Start simulated WebSocket in a background thread."""
        self._stop_event.clear()
        self.thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.thread.start()
        time.sleep(0.5)  # Brief wait for initialization

    def stop_websocket(self):
        """Stop the simulated WebSocket connection."""
        self._stop_event.set()
        self.status.connected = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def get_predicted_prices(self, token: str) -> Optional[float]:
        """Get predicted price for a token (returns current price for compatibility)."""
        if token in self.current_prices:
            return self.current_prices[token][0]
        return None

    def get_current_price(self, token: str) -> Optional[Tuple[float, str]]:
        """Get the current price for a token."""
        if token in self.current_prices:
            price, timestamp = self.current_prices[token]
            timestamp_iso = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()
            return (price, timestamp_iso)
        return None

    def get_historical_prices(self, token: str, count: int) -> Optional[np.ndarray]:
        """Get recent historical prices for a token."""
        if token in self.historical_data and self.historical_data[token]:
            return np.ndarray(self.historical_data[token][-count:])
        return None

    def load_historical_prices(
        self,
        token: str,
        days: int = 5,
        page_size: int = 1000,
        max_prices: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Generate simulated historical tick prices for a token at ~1s resolution.
        
        Creates realistic price movements matching real market tick data format.
        
        Args:
            token: The token symbol to generate data for
            days: Number of days of historical data to generate (default: 5)
            page_size: Unused (for interface compatibility)
            max_prices: Maximum number of tick prices to return
            
        Returns:
            np.ndarray of (price, timestamp) tuples at ~1s resolution
        """
        try:
            end_time = dt.datetime.now(dt.timezone.utc)
            start_time = end_time - dt.timedelta(days=days)
            
            # Check cache first
            cached_data = self._get_cached_prices(token, start_time, end_time)
            if cached_data is not None and len(cached_data) > 0:
                if max_prices is None or len(cached_data) >= max_prices:
                    result = cached_data[-max_prices:] if max_prices else cached_data
                    self.status.add_log('INFO',
                        f'Returning {len(result)} cached simulated ticks for {token}')
                    return np.array([
                        (p, ts.timestamp()) for p, ts in result
                    ])
            
            self.status.add_log('INFO',
                f'Generating simulated tick data for {token} ({days} days)...')
            
            # Calculate expected number of ticks
            timeframe_ms = self.TIMEFRAME_MS.get(self.timeframe, 1_000)
            total_ms = int((end_time - start_time).total_seconds() * 1000)
            expected_ticks = total_ms // timeframe_ms
            
            if max_prices:
                expected_ticks = min(expected_ticks, max_prices)
            
            # Generate tick data
            all_prices = self._generate_tick_prices(
                token,
                num_ticks=expected_ticks,
                start_time=start_time
            )
            
            # Update cache
            self._update_historical_cache(token, all_prices)
            
            self.status.add_log('INFO',
                f'Generated {len(all_prices)} simulated ticks for {token} '
                f'(from {all_prices[0][1]} to {all_prices[-1][1]})')
            
            # Convert to numpy array with float timestamps
            result = np.array([
                (float(p[0]), dt.datetime.fromisoformat(p[1].replace('Z', '+00:00')).timestamp())
                for p in all_prices
            ])
            
            return result
            
        except Exception as e:
            error_msg = f"Error generating simulated data for {token}: {e}"
            self.status.last_error = error_msg
            self.status.add_log('ERROR', error_msg)
            return None

    def close(self):
        """Close the simulated client and cleanup."""
        self.stop_websocket()
        self.status.running = False
        self.status.add_log('INFO', 'SimulatedPriceClient closed')


# Convenience factory functions
def create_simulated_client(
    tokens: List[str],
    base_prices: Optional[Dict[str, float]] = None,
    **kwargs
) -> SimulatedPriceClient:
    """Create a simulated price client with default parameters."""
    return SimulatedPriceClient(tokens=tokens, base_prices=base_prices, **kwargs)


def create_high_volatility_client(
    tokens: List[str],
    base_prices: Optional[Dict[str, float]] = None,
    **kwargs
) -> SimulatedPriceClient:
    """Create a simulated client with high volatility (120% annual)."""
    return SimulatedPriceClient(
        tokens=tokens,
        base_prices=base_prices,
        annual_volatility=1.2,
        **kwargs
    )


def create_low_volatility_client(
    tokens: List[str],
    base_prices: Optional[Dict[str, float]] = None,
    **kwargs
) -> SimulatedPriceClient:
    """Create a simulated client with low volatility (40% annual)."""
    return SimulatedPriceClient(
        tokens=tokens,
        base_prices=base_prices,
        annual_volatility=0.4,
        **kwargs
    )


# Usage example
if __name__ == "__main__":
    """
    Example usage - Simulated Tick Data:
    
    # Create simulated client with custom base prices
    client = SimulatedPriceClient(
        tokens=['BTC', 'ETH'],
        base_prices={'BTC': 100000, 'ETH': 3500},
        timeframe='1s',
        annual_volatility=0.80,  # 80% annualized volatility
        random_seed=42  # For reproducibility
    )
    
    # Generate 1 hour of tick data (3600 ticks at 1s resolution)
    btc_ticks = client.load_historical_prices('BTC', days=1, max_prices=3600)
    print(f"Generated {len(btc_ticks)} BTC ticks")
    print(f"Format: {btc_ticks.shape}")  # (3600, 2) - price, timestamp pairs
    
    # Verify tick spacing
    import numpy as np
    intervals = np.diff(btc_ticks[:, 1])
    print(f"Mean tick interval: {intervals.mean():.2f}s")  # Should be ~1.0s
    
    # Start live simulation
    client.start_background_websocket()
    
    # Get current price
    price, timestamp = client.get_current_price('BTC')
    print(f"BTC: ${price:.2f} at {timestamp}")
    
    # Use with PerpetualFuturesEnv
    # env = PerpetualFuturesEnv(tokens=['BTC'], price_client=client)
    
    # Cleanup
    client.close()
    """
    
    # Quick test
    print("Testing SimulatedPriceClient with 1s tick data...")
    
    client = SimulatedPriceClient(
        tokens=['BTC', 'ETH'],
        base_prices={'BTC': 100000, 'ETH': 3500},
        timeframe='1s',
        random_seed=42
    )
    
    # Generate 1 minute of tick data (60 ticks at 1s resolution)
    prices = client.load_historical_prices('BTC', days=1, max_prices=60)
    
    if prices is not None:
        print(f"Generated {len(prices)} ticks")
        print(f"Shape: {prices.shape}")
        print(f"First tick: price=${prices[0][0]:.2f}, ts={prices[0][1]:.0f}")
        print(f"Last tick: price=${prices[-1][0]:.2f}, ts={prices[-1][1]:.0f}")
        
        # Verify 1s spacing
        intervals = np.diff(prices[:, 1])
        print(f"Tick interval: {intervals.mean():.2f}s (should be 1.0s)")
        
        # Show price statistics
        price_returns = np.diff(np.log(prices[:, 0]))
        print(f"Return mean: {price_returns.mean():.6f}")
        print(f"Return std: {price_returns.std():.6f}")
    else:
        print("Failed to generate tick data")
    
    client.close()