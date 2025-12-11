import asyncio
import websockets
import json
import threading
import time
import datetime as dt
import numpy as np
import tqdm
from typing import List, Optional, Tuple
import requests

class PriceServerClient:
    """Client to interact with the price server via WebSocket and REST API"""
    
    def __init__(
        self, 
        tokens: List[str], 
        rest_api_url: str, 
        websocket_url: str, 
        timezone_offset: str = '-00:00'
    ): #Timezone offset added
        from . import PriceClientStatus

        self.tokens = tokens
        self.rest_api_url = rest_api_url
        self.websocket_url = websocket_url
        self.current_prices = {token: None for token in tokens}
        # Historical data cache: {token: {'timestamps': list, 'prices': list, 'min_time': datetime, 'max_time': datetime}}
        self._historical_cache = {token: {'timestamps': [], 'prices': [], 'min_time': None, 'max_time': None} for token in tokens}
        # For backward compatibility
        self.historical_data = {token: [] for token in tokens}
        self.websocket = None
        self.connected = False
        self.loop = None
        self.thread = None
        self.timezone_offset = timezone_offset #Timezone offset saved
        self.status = PriceClientStatus()     
        self.status.logs = {"INFO": [], "WARNING": [], "ERROR": [], "DEBUG": []}   
        self.status.running = True
        self.status.start_time = time.time()
        self.status.add_log('INFO', 'Service initialized')
        
    def _update_historical_cache(self, token: str, new_prices: List[Tuple[float, str]]):
        """
        Update the historical cache with new price data, avoiding duplicates.
        
        Args:
            token: The token symbol
            new_prices: List of (price, timestamp) tuples
        """
        if not new_prices or token not in self._historical_cache:
            return
            
        cache = self._historical_cache[token]
        
        # Convert timestamps to datetime objects for comparison
        price_objs = []
        for price, ts_str in new_prices:
            try:
                # Parse the timestamp string to datetime object
                try:
                    # First try parsing with fromisoformat which handles most cases
                    if 'T' in ts_str:
                        # Handle ISO 8601 format with timezone offset
                        if '+' in ts_str or '-' in ts_str.split('T')[-1] or ts_str.endswith('Z'):
                            # Has timezone info
                            ts_str_clean = ts_str.replace('Z', '+00:00')
                            # Handle microseconds if present
                            if '.' in ts_str_clean:
                                # Parse with microseconds
                                ts = dt.datetime.strptime(ts_str_clean, '%Y-%m-%dT%H:%M:%S.%f%z')
                            else:
                                # Parse without microseconds
                                ts = dt.datetime.strptime(ts_str_clean, '%Y-%m-%dT%H:%M:%S%z')
                        else:
                            # No timezone info, assume UTC
                            if '.' in ts_str:
                                # With microseconds
                                ts = dt.datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S.%f')
                            else:
                                # Without microseconds
                                ts = dt.datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%S')
                            ts = ts.replace(tzinfo=dt.timezone.utc)
                    else:
                        # Handle other formats if needed
                        ts = dt.datetime.fromisoformat(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=dt.timezone.utc)
                except (ValueError, TypeError) as e:
                    # Fallback to more flexible parsing if standard formats fail
                    try:
                        # Try parsing with dateutil.parser if available
                        from dateutil import parser
                        ts = parser.parse(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.replace(tzinfo=dt.timezone.utc)
                    except (ImportError, ValueError):
                        # If all else fails, log the error and skip this entry
                        self.status.add_log('WARNING', f'Failed to parse timestamp {ts_str}: {e}')
                        continue
                
                price_objs.append((price, ts))
            except (ValueError, TypeError) as e:
                self.status.add_log('WARNING', f'Failed to parse timestamp {ts_str}: {e}')
        
        if not price_objs:
            return
            
        # Sort new prices by timestamp
        price_objs.sort(key=lambda x: x[1])
        
        # Update cache
        if not cache['timestamps']:
            # Initialize cache with first batch of data
            cache['prices'] = [p[0] for p in price_objs]
            cache['timestamps'] = [p[1] for p in price_objs]
            cache['min_time'] = price_objs[0][1]
            cache['max_time'] = price_objs[-1][1]
        else:
            # Merge new data with existing cache
            all_prices = list(zip(cache['prices'], cache['timestamps']))
            all_prices.extend(price_objs)
            
            # Remove duplicates (keeping the last occurrence)
            seen = {}
            for price, ts in all_prices:
                seen[ts] = price
                
            # Sort by timestamp
            sorted_items = sorted(seen.items(), key=lambda x: x[0])
            
            # Update cache
            cache['timestamps'] = [item[0] for item in sorted_items]
            cache['prices'] = [item[1] for item in sorted_items]
            cache['min_time'] = cache['timestamps'][0]
            cache['max_time'] = cache['timestamps'][-1]
        
        # Update the legacy historical_data for backward compatibility
        self.historical_data[token] = list(zip(cache['prices'], [ts.isoformat() for ts in cache['timestamps']]))
        
        self.status.add_log('DEBUG', 
            f'Updated cache for {token} with {len(price_objs)} new prices. '
            f'Cache now contains {len(cache["prices"])} prices from {cache["min_time"]} to {cache["max_time"]}'
        )
    
    def _get_cached_prices(self, token: str, start_time: dt.datetime, end_time: dt.datetime) -> Optional[List[Tuple[float, str]]]:
        """
        Get prices from cache for the specified time range.
        
        Args:
            token: The token symbol
            start_time: Start of time range (inclusive)
            end_time: End of time range (inclusive)
            
        Returns:
            List of (price, timestamp) tuples if cache has data for the full range, None otherwise
        """
        if token not in self._historical_cache:
            return None
            
        cache = self._historical_cache[token]
        
        # If cache is empty, nothing to return
        if not cache['timestamps']:
            return None
            
        # If requested range is entirely outside cache, return None
        if (cache['max_time'] < start_time) or (cache['min_time'] > end_time):
            return None
            
        # Find indices of timestamps within the requested range
        start_idx = 0
        end_idx = len(cache['timestamps'])
        
        # Binary search for start time
        left, right = 0, len(cache['timestamps'])
        while left < right:
            mid = (left + right) // 2
            if cache['timestamps'][mid] < start_time:
                left = mid + 1
            else:
                right = mid
        start_idx = left
        
        # Binary search for end time
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
            
        # Return prices and timestamps in the range
        result = [
            (cache['prices'][i], cache['timestamps'][i])
            for i in range(start_idx, end_idx)
        ]
        
        self.status.add_log('DEBUG', 
            f'Cache hit for {token} from {start_time} to {end_time}: '
            f'found {len(result)} prices in cache.'
        )
        
        return result
        
    async def connect_websocket(self):
        """Connect to the WebSocket server"""
        try:
            for token in self.tokens:
                url = f"{self.websocket_url}/{token}"
                self.websocket = await websockets.connect(url)
                self.status.connected = True
                self.status.add_log('INFO', f"Connected to WebSocket server for {token}")
            
                # Subscribe to price updates for all tokens
                #subscribe_msg = {
                #    "action": "subscribe",
                #    "tokens": self.tokens
                #}
                #await self.websocket.send(json.dumps(subscribe_msg))
            
                # Start listening for messages
                await self.listen(token)
        except Exception as e:
            self.status.last_error = f"WebSocket connection failed: {e}"
            self.status.add_log('ERROR', f"WebSocket connection failed: {e}")
            self.status.connected = False
            
    async def listen(self, token: str):
        """Listen for incoming WebSocket messages"""
        while self.status.connected:
            try:
                message = await self.websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'price_update':
                    #token = data['token'] #NO TOKEN IN MESSAGE
                    price = float(data['data']['price'])
                    timestamp = data['data']['timestamp']
                    self.current_prices[token] = (price, timestamp)
                    self.historical_data[token].append((price, timestamp))
                    self.status.add_log('DEBUG', f"Received price update for {token}: {price} at {timestamp}")
                elif data.get('type') == 'ping':
                    self.status.add_log('DEBUG', "Received ping from server")
                else:
                    self.status.add_log('WARNING', f"Received unknown message type: {data.get('type')}")
                    
            except websockets.exceptions.ConnectionClosed as e:
                self.status.last_error = f"WebSocket connection closed for {token}: {e}"
                self.status.add_log('ERROR', f"WebSocket connection closed for {token}: {e}")
                self.status.connected = False
                break
            except Exception as e:
                self.status.last_error = f"Error processing WebSocket message for {token}: {e}"
                self.status.add_log('ERROR', f"Error processing WebSocket message for {token}: {e}")

    def start_websocket(self):
        """Start the WebSocket connection in a separate thread"""
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
        """Start WebSocket in a background thread"""
        self.thread = threading.Thread(target=self.start_websocket, daemon=True)
        self.thread.start()
        
        # Wait a bit for connection to establish
        time.sleep(2)

    def get_predicted_prices(self, token: str) -> Optional[np.ndarray]:
        """Get predicted prices for a token"""
        if token in self.current_prices and self.current_prices[token] is not None:
            return self.current_prices[token][0]
        return None

    def get_current_price(self, token: str) -> Optional[float]:
        """Get the current price for a token"""
        if token in self.current_prices and self.current_prices[token] is not None:
            return self.current_prices[token][0]
        return None

    def get_historical_prices(self, token: str, count: int) -> Optional[np.ndarray]:
        """Get historical prices for a token"""
        if token in self.historical_data and self.historical_data[token] is not None:
            return np.array(self.historical_data[token][-count:])
        return None
        
    def _pull_historical_prices(self, token: str, start: dt.datetime, end: dt.datetime, page: int = 1, page_size: int = 5000):
        # Format the timestamps as required by the API (YYYY-MM-DDTHH:MM:SS-TZ_OFFSET)
        start_str = start.isoformat().split('+')[0] + self.timezone_offset
        end_str = end.isoformat().split('+')[0] + self.timezone_offset

        endpoint = f"{self.rest_api_url}/historic/price/{token}"

        params = {
            "start": start_str,
            "end": end_str,
            "page": page,
            "page_size": page_size
        }

        headers = {
            'ngrok-skip-browser-warning': 'true'
        }

        # Make the GET request
        response = requests.get(endpoint, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()

        return data

    def _parse_historical_prices(self, data: dict) -> np.ndarray[Tuple[float, str]]:
        return np.array([
            (float(item['price']), item['timestamp'])
            for item in data.get('data', [])
        ])

    def load_historical_prices(
        self, 
        token: str, 
        days: int = 5, 
        page_size: int = 5000,
        max_prices: Optional[int] = None
    ) -> Optional[np.ndarray[Tuple[float, float]]]:
        """
        Get historical prices for a token using the REST API with pagination.
        Uses an intelligent cache to minimize API calls.
        
        Args:
            token: The token symbol to fetch data for
            days: Number of days of historical data to fetch (default: 5)
            page_size: Number of items per page (default: 1000)
            max_prices: Maximum number of prices to return. If None, returns all available prices.
                      If specified, returns the most recent prices up to this count.
            
        Returns:
            Optional[np.ndarray]: Array of historical prices or None if an error occurs.
                               If max_prices is specified, returns at most max_prices prices.
        """
        try:
            # Define start and end time
            end_time = dt.datetime.now(dt.timezone.utc)
            start_time = end_time - dt.timedelta(days=days)
            
            # First, try to get data from cache
            cached_data = self._get_cached_prices(token, start_time, end_time)
            
            # If we have all the data in cache, return it
            if cached_data is not None and len(cached_data) > 0:
                if max_prices is not None and len(cached_data) >= max_prices:
                    max_pages = max_prices // page_size
                    cached_data = cached_data[-max_pages:]
                self.status.add_log('INFO', 
                    f'Returning {len(cached_data)} cached prices for {token} from {start_time} to {end_time}.'
                    f'Requested max: {max_prices}.'
                )
                return np.array(cached_data)
                
            # If we get here, we need to fetch from the API
            self.status.add_log('INFO', 
                f'Cache miss for {token} from {start_time} to {end_time}. Fetching from API...'
            )
            
            
            
            all_prices = []
            page = 1
            
            data = self._pull_historical_prices(token, start_time, end_time, page, page_size)
            
            # Extract the prices from the current page
            page_prices = self._parse_historical_prices(data)
            
            # If max_prices is specified, only take as many as we need
            if max_prices is not None:
                remaining = max_prices - len(all_prices)
                if remaining <= 0:
                    return np.array(all_prices)
                page_prices = page_prices[:remaining]
            
            all_prices.extend(page_prices)

            total_pages = data.get('total_pages', 1)
            total_items = data.get('total', 0)
            
            # Initialize tqdm progress bar
            progress_bar = tqdm.tqdm(
                total=min(total_items, max_prices) if max_prices is not None else total_items,
                desc=f"Loading {token} prices",
                unit=" prices",
                ncols=100,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [Time: {elapsed}<{remaining}, {rate_fmt}]',
                leave=False
            )
            
            self.status.add_log('INFO', 
                f"Fetching {total_items} historical prices for {token} "
                f"({total_pages} pages of {page_size} items)"
            )
            
            while page <= total_pages:
                # Construct the URL with pagination parameters
                data = self._pull_historical_prices(token, start_time, end_time, page, page_size)
                
                # Extract the prices from the current page
                page_prices = self._parse_historical_prices(data)
                
                # If max_prices is specified, only take as many as we need
                if max_prices is not None:
                    remaining = max_prices - len(all_prices)
                    if remaining <= 0:
                        break
                    page_prices = page_prices[:remaining]
                
                all_prices.extend(page_prices)
                
                # Log progress
                self.status.add_log('DEBUG', 
                    f"Fetched page {page}/{total_pages} for {token}: "
                    f"{len(page_prices)} items (total: {len(all_prices)}"
                    f"{f' (max: {max_prices})' if max_prices is not None else ''})"
                )
                
                # Update progress bar
                progress_bar.update(len(page_prices))
                
                # Check if we've reached the max_prices limit
                if max_prices is not None and len(all_prices) >= max_prices:
                    all_prices = all_prices[:max_prices]
                    progress_bar.n = min(progress_bar.n, max_prices)
                    progress_bar.refresh()
                    break
                    
                # Move to next page if we haven't reached the max_prices limit
                page += 1
                
                # Small delay between requests to avoid overwhelming the server
                if page <= total_pages:
                    time.sleep(0.1)
            
            # Close the progress bar
            progress_bar.close()
            
            if not all_prices:
                self.status.add_log('WARNING', f"No historical price data found for {token}")
                return None
                
            # Update the cache with the newly fetched data
            self._update_historical_cache(token, all_prices)
            
            # Return the requested number of prices (most recent first)
            if max_prices is not None and len(all_prices) > max_prices:
                all_prices = all_prices[-max_prices:]
                
            self.status.add_log('INFO', 
                f"Successfully fetched {len(all_prices)} historical prices for {token} "
                f"(from {all_prices[0][1]} to {all_prices[-1][1]})"
            )

            all_prices = np.array([(p[0], dt.datetime.fromisoformat(p[1]).timestamp()) for p in all_prices])
            
            return all_prices

        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching historical data for {token}: {e}"
            self.status.last_error = error_msg
            self.status.add_log('ERROR', error_msg)
            return None
            
        except (KeyError, TypeError) as e:
            error_msg = f"Error parsing historical data for {token}: {e}"
            self.status.last_error = error_msg
            self.status.add_log('ERROR', error_msg)
            return None
            
        except Exception as e:
            error_msg = f"Unexpected error fetching historical data for {token}: {e}"
            self.status.last_error = error_msg
            self.status.add_log('ERROR', error_msg)
            return None