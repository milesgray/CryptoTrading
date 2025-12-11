"""
Exchange Price Client using CCXT library.

Pulls historic data or listens to real-time price updates from cryptocurrency 
exchanges via the ccxt library. Matches the PriceServerClient interface.
"""

import asyncio
import threading
import time
import datetime as dt
import numpy as np
import logging
import tqdm
from typing import List, Optional, Tuple, Dict

from cryptotrading.util.ccxt import pull_ohlcv_data

try:
    import ccxt
    import ccxt.pro as ccxtpro
    CCXT_PRO_AVAILABLE = True
except ImportError:
    import ccxt
    CCXT_PRO_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExchangePriceClient:
    """
    Client to fetch tick-level price data from cryptocurrency exchanges via CCXT.
    
    Returns price data in tick format at ~1 second resolution, matching PriceServerClient.
    Supports both REST API for historical data and WebSocket for real-time updates.
    
    Data Resolution:
        - Default: 1-second ticks (one price point per second)
        - Binance: Uses native 1s candle close prices
        - Other exchanges: Aggregates trade data into 1s ticks
    
    Args:
        tokens: List of token symbols (e.g., ['BTC', 'ETH']). Will be converted to 
                exchange pairs like 'BTC/USDT'.
        exchange_id: CCXT exchange identifier (default: 'binance'). Options include
                     'binance', 'coinbase', 'kraken', 'bybit', 'okx', etc.
        quote_currency: Quote currency for pairs (default: 'USDT').
        timeframe: Data resolution (default: '1s' for 1-second ticks).
        sandbox: Whether to use exchange sandbox/testnet (default: False).
        api_key: Optional API key for authenticated endpoints.
        api_secret: Optional API secret for authenticated endpoints.
    
    Example:
        >>> client = ExchangePriceClient(tokens=['BTC'])
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
    
    # Exchanges that support 1s candles natively
    EXCHANGES_WITH_1S_CANDLES = {'binance', 'binanceusdm', 'binancecoinm'}
    
    def __init__(
        self,
        tokens: List[str],
        exchange_id: str = 'binanceus',
        quote_currency: str = 'USDT',
        timeframe: str = '1s',
        sandbox: bool = False,
        api_key: Optional[str] = None,
        api_secret: Optional[str] = None,
    ):
        from . import PriceClientStatus
        
        self.tokens = tokens
        self.exchange_id = exchange_id
        self.quote_currency = quote_currency
        self.timeframe = timeframe
        self.sandbox = sandbox
        
        # Convert tokens to exchange symbols (e.g., 'BTC' -> 'BTC/USDT')
        self.symbols = {token: f"{token}/{quote_currency}" for token in tokens}
        
        # Initialize exchange
        exchange_class = getattr(ccxt, exchange_id)
        exchange_config = {
            'enableRateLimit': True,
            'options': {'defaultType': 'future'},  # Use futures market
        }
        
        if api_key and api_secret:
            exchange_config['apiKey'] = api_key
            exchange_config['secret'] = api_secret
            
        self.exchange = exchange_class(exchange_config)
        
        if sandbox:
            self.exchange.set_sandbox_mode(True)
            
        # Initialize async exchange for WebSocket (if ccxt.pro available)
        self.async_exchange = None
        if CCXT_PRO_AVAILABLE:
            async_exchange_class = getattr(ccxtpro, exchange_id, None)
            if async_exchange_class:
                self.async_exchange = async_exchange_class(exchange_config)
                if sandbox:
                    self.async_exchange.set_sandbox_mode(True)
        
        # Price storage - matches PriceServerClient interface
        self.current_prices: Dict[str, Optional[Tuple[float, str]]] = {
            token: None for token in tokens
        }
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
        
        # Connection state
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
        self.status.add_log('INFO', f'ExchangePriceClient initialized for {exchange_id}')
        
        # Markets loaded flag (lazy loading)
        self._markets_loaded = False

    def _ensure_markets_loaded(self):
        """Lazy load exchange markets if not already loaded."""
        if self._markets_loaded:
            return True
        try:
            self.exchange.load_markets()
            self._markets_loaded = True
            market_count = len(self.exchange.markets) if self.exchange.markets else 0
            self.status.add_log('INFO', f'Loaded {market_count} markets from {self.exchange_id}')
            return True
        except Exception as e:
            self.status.add_log('WARNING', f'Could not load markets (may still work): {e}')
            return False

    def _symbol_for_token(self, token: str) -> str:
        """Get the exchange symbol for a token."""
        return self.symbols.get(token, f"{token}/{self.quote_currency}")

    def _update_historical_cache(self, token: str, new_prices: List[Tuple[float, str]]):
        """
        Update the historical cache with new price data, avoiding duplicates.
        
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
                    # Unix timestamp
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
        
        self.status.add_log('DEBUG',
            f'Updated cache for {token} with {len(price_objs)} prices. '
            f'Cache now contains {len(cache["prices"])} prices.'
        )

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
        
        self.status.add_log('DEBUG',
            f'Cache hit for {token}: found {len(result)} prices.'
        )
        
        return result

    async def connect_websocket(self):
        """Connect to exchange WebSocket for real-time price updates."""
        if not CCXT_PRO_AVAILABLE or self.async_exchange is None:
            self.status.add_log('WARNING', 
                'ccxt.pro not available. Using polling fallback for real-time data.')
            await self._polling_fallback()
            return
            
        try:
            self.status.connected = True
            self.status.add_log('INFO', f'Connecting to {self.exchange_id} WebSocket...')
            
            while not self._stop_event.is_set() and self.status.connected:
                try:
                    for token in self.tokens:
                        symbol = self._symbol_for_token(token)
                        
                        # Watch ticker for real-time price
                        ticker = await self.async_exchange.watch_ticker(symbol)
                        
                        if ticker and 'last' in ticker:
                            price = float(ticker['last'])
                            timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                            
                            self.current_prices[token] = (price, timestamp)
                            self.historical_data[token].append((price, timestamp))
                            
                            # Trim historical data to prevent memory bloat
                            if len(self.historical_data[token]) > 10000:
                                self.historical_data[token] = self.historical_data[token][-5000:]
                            
                            self.status.add_log('DEBUG',
                                f'Received price for {token}: {price} at {timestamp}')
                                
                except ccxt.NetworkError as e:
                    self.status.add_log('WARNING', f'Network error: {e}. Reconnecting...')
                    await asyncio.sleep(5)
                except Exception as e:
                    self.status.add_log('ERROR', f'WebSocket error: {e}')
                    await asyncio.sleep(1)
                    
        except Exception as e:
            self.status.last_error = f"WebSocket connection failed: {e}"
            self.status.add_log('ERROR', f"WebSocket connection failed: {e}")
            self.status.connected = False
        finally:
            if self.async_exchange:
                await self.async_exchange.close()

    async def _polling_fallback(self):
        """Fallback to REST polling when WebSocket unavailable."""
        self.status.connected = True
        self.status.add_log('INFO', 'Using REST polling for price updates...')
        
        while not self._stop_event.is_set() and self.status.connected:
            try:
                for token in self.tokens:
                    symbol = self._symbol_for_token(token)
                    ticker = self.exchange.fetch_ticker(symbol)
                    
                    if ticker and 'last' in ticker:
                        price = float(ticker['last'])
                        timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                        
                        self.current_prices[token] = (price, timestamp)
                        self.historical_data[token].append((price, timestamp))
                        
                        if len(self.historical_data[token]) > 10000:
                            self.historical_data[token] = self.historical_data[token][-5000:]
                
                await asyncio.sleep(1)  # Poll every second
                
            except Exception as e:
                self.status.add_log('ERROR', f'Polling error: {e}')
                await asyncio.sleep(5)

    def start_websocket(self):
        """Start the WebSocket connection in the current thread."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self.connect_websocket())
        except Exception as e:
            self.status.last_error = f"Error running websocket loop: {e}"
            self.status.add_log('ERROR', f"Error running websocket loop: {e}")
        finally:
            self.loop.close()

    def start_background_websocket(self):
        """Start WebSocket in a background thread."""
        self._stop_event.clear()
        self.thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.thread.start()
        time.sleep(2)  # Wait for connection

    def stop_websocket(self):
        """Stop the WebSocket connection."""
        self._stop_event.set()
        self.status.connected = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

    def get_predicted_prices(self, token: str) -> Optional[np.ndarray]:
        """Get predicted prices for a token (returns current price for compatibility)."""
        if token in self.current_prices and self.current_prices[token] is not None:
            return self.current_prices[token][0]
        return None

    def get_current_price(self, token: str) -> Optional[Tuple[float, str]]:
        """Get the current price for a token."""
        if token in self.current_prices and self.current_prices[token] is not None:
            return self.current_prices[token]
        
        # Fetch from exchange if no cached price
        self._ensure_markets_loaded()
        try:
            symbol = self._symbol_for_token(token)
            ticker = self.exchange.fetch_ticker(symbol)
            if ticker and 'last' in ticker:
                price = float(ticker['last'])
                timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
                self.current_prices[token] = (price, timestamp)
                return (price, timestamp)
        except Exception as e:
            self.status.add_log('ERROR', f'Failed to fetch current price for {token}: {e}')
        
        return None

    def get_historical_prices(self, token: str, count: int) -> Optional[np.ndarray]:
        """Get recent historical prices for a token."""
        if token in self.historical_data and self.historical_data[token]:
            return np.array(self.historical_data[token][-count:])
        return None

    def _supports_1s_candles(self) -> bool:
        """Check if current exchange supports 1s candles natively."""
        return self.exchange_id in self.EXCHANGES_WITH_1S_CANDLES

    def _fetch_ohlcv(
        self, 
        token: str, 
        since: int, 
        limit: int = 1000
    ) -> List[Tuple[float, str]]:
        """
        Fetch OHLCV data from exchange.
        
        Args:
            token: Token symbol
            since: Unix timestamp in milliseconds
            limit: Number of candles to fetch
            
        Returns:
            List of (close_price, timestamp_iso) tuples
        """
        self._ensure_markets_loaded()
        
        try:
            results = pull_ohlcv_data(token, n=1, span="1s", min_uts=since/1000, exchange=self.exchange_id)

            results = results[:limit]
            return results
        except ccxt.RateLimitExceeded:
            self.status.add_log('WARNING', 'Rate limit hit, waiting...')
            time.sleep(self.exchange.rateLimit / 1000)
            return self._fetch_ohlcv(token, since, limit)
            
        except Exception as e:
            self.status.add_log('ERROR', f'Failed to fetch OHLCV for {token}: {e}')
            return []

    def _fetch_trades_as_ticks(
        self,
        token: str,
        since: int,
        limit: int = 1000
    ) -> List[Tuple[float, str]]:
        """
        Fetch raw trades and aggregate to 1-second ticks.
        
        Used for exchanges that don't support 1s candles natively.
        Takes the last trade price in each 1-second bucket.
        
        Args:
            token: Token symbol
            since: Unix timestamp in milliseconds
            limit: Approximate number of 1s ticks to return
            
        Returns:
            List of (price, timestamp_iso) tuples at 1s resolution
        """
        symbol = self._symbol_for_token(token)
        self._ensure_markets_loaded()
        
        try:
            # Fetch more trades than needed since we'll aggregate
            # Typically need ~10-50 trades per second for liquid pairs
            trades_limit = min(limit * 20, 1000)  # Most exchanges cap at 1000
            
            all_trades = []
            current_since = since
            target_ticks = limit
            
            while len(all_trades) < trades_limit * 2 and target_ticks > 0:
                trades = self.exchange.fetch_trades(
                    symbol,
                    since=current_since,
                    limit=trades_limit
                )
                
                if not trades:
                    break
                    
                all_trades.extend(trades)
                
                # Move forward in time
                last_trade_ts = trades[-1]['timestamp']
                if last_trade_ts <= current_since:
                    break
                current_since = last_trade_ts + 1
                
                # Check if we have enough seconds covered
                if len(all_trades) > 0:
                    time_span_ms = all_trades[-1]['timestamp'] - all_trades[0]['timestamp']
                    if time_span_ms >= target_ticks * 1000:
                        break
                
                time.sleep(self.exchange.rateLimit / 1000)
            
            if not all_trades:
                return []
            
            # Aggregate trades into 1-second buckets
            # Use the LAST trade price in each second (closing price)
            tick_buckets: Dict[int, Tuple[float, int]] = {}  # second -> (price, timestamp_ms)
            
            for trade in all_trades:
                ts_ms = trade['timestamp']
                price = float(trade['price'])
                second_bucket = ts_ms // 1000
                
                # Keep the latest trade in each bucket
                if second_bucket not in tick_buckets or ts_ms > tick_buckets[second_bucket][1]:
                    tick_buckets[second_bucket] = (price, ts_ms)
            
            # Sort by timestamp and convert to output format
            sorted_seconds = sorted(tick_buckets.keys())
            prices = []
            
            for second in sorted_seconds:
                price, _ = tick_buckets[second]
                ts_dt = dt.datetime.fromtimestamp(second, tz=dt.timezone.utc)
                prices.append((price, ts_dt.isoformat()))
            
            self.status.add_log('DEBUG',
                f'Aggregated {len(all_trades)} trades into {len(prices)} 1s ticks for {token}')
            
            return prices[:limit]
            
        except ccxt.RateLimitExceeded:
            self.status.add_log('WARNING', 'Rate limit hit on trades fetch, waiting...')
            time.sleep(self.exchange.rateLimit / 1000 * 2)
            return self._fetch_trades_as_ticks(token, since, limit)
            
        except Exception as e:
            self.status.add_log('ERROR', f'Failed to fetch trades for {token}: {e}')
            return []

    def _fetch_tick_data(
        self,
        token: str,
        since: int,
        limit: int = 1000
    ) -> List[Tuple[float, str]]:
        """
        Fetch tick data at ~1s resolution.
        
        Uses 1s candles if supported by exchange, otherwise aggregates trades.
        
        Args:
            token: Token symbol
            since: Unix timestamp in milliseconds
            limit: Number of ticks to fetch
            
        Returns:
            List of (price, timestamp_iso) tuples
        """
        # If requesting 1s resolution and exchange supports it, use OHLCV
        if self.timeframe == '1s' and self._supports_1s_candles():
            return self._fetch_ohlcv(token, since, limit)
        
        # If requesting 1s but exchange doesn't support, use trades
        if self.timeframe == '1s':
            return self._fetch_trades_as_ticks(token, since, limit)
        
        # Otherwise use standard OHLCV for the requested timeframe
        return self._fetch_ohlcv(token, since, limit)

    def load_historical_prices(
        self,
        token: str,
        days: int = 5,
        page_size: int = 1000,
        max_prices: Optional[int] = None
    ) -> Optional[np.ndarray]:
        """
        Get historical tick prices for a token at ~1s resolution.
        
        Returns price data in tick format matching PriceServerClient:
        - Default 1s resolution (one price per second)
        - Uses 1s candle close prices on Binance
        - Aggregates trades to 1s ticks on other exchanges
        
        Args:
            token: The token symbol to fetch data for
            days: Number of days of historical data to fetch (default: 5)
            page_size: Number of ticks per request (default: 1000)
            max_prices: Maximum number of tick prices to return
            
        Returns:
            np.ndarray of (price, timestamp) tuples at ~1s resolution, or None on error
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
                        f'Returning {len(result)} cached tick prices for {token}')
                    return np.array([
                        (p, ts.timestamp()) for p, ts in result
                    ])
            
            # Log fetch method being used
            if self.timeframe == '1s':
                if self._supports_1s_candles():
                    method = "1s candles"
                else:
                    method = "trade aggregation"
            else:
                method = f"{self.timeframe} candles"
                
            self.status.add_log('INFO',
                f'Fetching {token} tick data from {self.exchange_id} via {method}...')
            
            # Calculate expected number of ticks
            timeframe_ms = self.TIMEFRAME_MS.get(self.timeframe, 1_000)
            total_ms = int((end_time - start_time).total_seconds() * 1000)
            expected_ticks = total_ms // timeframe_ms
            
            if max_prices:
                expected_ticks = min(expected_ticks, max_prices)
            
            all_prices = []
            since_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            # Progress bar
            progress_bar = tqdm.tqdm(
                total=expected_ticks,
                desc=f"Loading {token} ticks from {self.exchange_id}",
                unit=" ticks",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
                leave=False
            )
            
            while since_ms < end_ms:
                # Respect max_prices limit
                if max_prices and len(all_prices) >= max_prices:
                    break
                
                batch_limit = min(page_size, max_prices - len(all_prices) if max_prices else page_size)
                
                # Use tick-aware fetching
                prices = self._fetch_tick_data(token, since_ms, batch_limit)
                
                if not prices:
                    break
                    
                all_prices.extend(prices)
                progress_bar.update(len(prices))
                
                # Move to next batch
                last_ts_str = prices[-1][1]
                last_ts = dt.datetime.fromisoformat(last_ts_str.replace('Z', '+00:00'))
                since_ms = int(last_ts.timestamp() * 1000) + timeframe_ms
                
                # Rate limiting
                time.sleep(self.exchange.rateLimit / 1000)
            
            progress_bar.close()
            
            if not all_prices:
                self.status.add_log('WARNING', f'No price data found for {token}')
                return None
            
            # Update cache
            self._update_historical_cache(token, all_prices)
            
            # Apply max_prices limit
            if max_prices and len(all_prices) > max_prices:
                all_prices = all_prices[-max_prices:]
            
            self.status.add_log('INFO',
                f'Loaded {len(all_prices)} tick prices for {token} '
                f'(from {all_prices[0][1]} to {all_prices[-1][1]})')
            
            # Convert to numpy array with float timestamps
            result = np.array([
                (float(p[0]), dt.datetime.fromisoformat(p[1].replace('Z', '+00:00')).timestamp())
                for p in all_prices
            ])
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching tick data for {token}: {e}"
            self.status.last_error = error_msg
            self.status.add_log('ERROR', error_msg)
            return None

    def fetch_funding_rate(self, token: str) -> Optional[float]:
        """
        Fetch current funding rate for a perpetual futures contract.
        
        Args:
            token: Token symbol
            
        Returns:
            Current funding rate or None
        """
        try:
            symbol = self._symbol_for_token(token)
            
            # Different exchanges have different methods
            if hasattr(self.exchange, 'fetch_funding_rate'):
                funding = self.exchange.fetch_funding_rate(symbol)
                return float(funding.get('fundingRate', 0))
            elif hasattr(self.exchange, 'fetch_premium_index_klines'):
                # Binance-specific
                funding = self.exchange.fetch_premium_index_klines(symbol, '8h', limit=1)
                if funding:
                    return float(funding[0][4])  # Close price as proxy
                    
        except Exception as e:
            self.status.add_log('WARNING', f'Failed to fetch funding rate for {token}: {e}')
        
        return None

    def fetch_orderbook(self, token: str, limit: int = 20) -> Optional[dict]:
        """
        Fetch current orderbook for a token.
        
        Args:
            token: Token symbol
            limit: Depth of orderbook
            
        Returns:
            Orderbook dict with 'bids' and 'asks'
        """
        try:
            symbol = self._symbol_for_token(token)
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
        except Exception as e:
            self.status.add_log('ERROR', f'Failed to fetch orderbook for {token}: {e}')
            return None

    def get_available_symbols(self) -> List[str]:
        """Get list of available trading symbols on the exchange."""
        self._ensure_markets_loaded()
        try:
            if self.exchange.markets:
                return list(self.exchange.markets.keys())
            return []
        except Exception:
            return []

    def close(self):
        """Close all connections and cleanup."""
        self.stop_websocket()
        try:
            self.exchange.close()
        except Exception:
            pass
        self.status.running = False
        self.status.add_log('INFO', 'ExchangePriceClient closed')


# Convenience factory functions for common exchanges
def create_binance_client(tokens: List[str], **kwargs) -> ExchangePriceClient:
    """Create a client for Binance exchange."""
    return ExchangePriceClient(tokens=tokens, exchange_id='binance', **kwargs)


def create_bybit_client(tokens: List[str], **kwargs) -> ExchangePriceClient:
    """Create a client for Bybit exchange."""
    return ExchangePriceClient(tokens=tokens, exchange_id='bybit', **kwargs)


def create_okx_client(tokens: List[str], **kwargs) -> ExchangePriceClient:
    """Create a client for OKX exchange."""
    return ExchangePriceClient(tokens=tokens, exchange_id='okx', **kwargs)