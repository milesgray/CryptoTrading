from typing import Optional
import numpy as np
import datetime as dt
from datetime import datetime, timedelta

class PriceLevels:
    """
    A class to detect, track and analyze price levels in financial data.
    Supports multiple timeframes and calculates level strength.
    """
    
    def __init__(self, lookback_periods=None, timeframes=None, level_threshold=0.05, 
                 level_proximity=0.01, max_levels=10):
        """
        Initialize the PriceLevels detector.
        
        Args:
            lookback_periods (dict): Dictionary mapping timeframe names to number of periods to look back
            timeframes (dict): Dictionary mapping timeframe names to timedelta objects
            level_threshold (float): Minimum touch count for a level to be considered valid (as % of data points)
            level_proximity (float): How close prices need to be to be considered the same level (as % of price)
            max_levels (int): Maximum number of levels to track
        """
        # Default lookback periods if not provided
        self.lookback_periods = lookback_periods or {
            '1m': 1440,    # 1 day of 1-minute data
            '5m': 1152,    # 4 days of 5-minute data
            '15m': 672,    # 7 days of 15-minute data
            '1h': 168,     # 7 days of hourly data
            '4h': 90,      # 15 days of 4-hour data
            '1d': 90,      # 90 days of daily data
        }
        
        # Default timeframes if not provided
        self.timeframes = timeframes or {
            '1m': timedelta(minutes=1),
            '5m': timedelta(minutes=5),
            '15m': timedelta(minutes=15),
            '1h': timedelta(hours=1),
            '4h': timedelta(hours=4),
            '1d': timedelta(days=1),
        }
        
        # Storage for raw price data
        self.raw_data = []
        
        # Storage for OHLC candles by timeframe
        self.candles = {tf: [] for tf in self.timeframes}
        
        # Storage for detected levels by timeframe
        self.levels = {tf: [] for tf in self.timeframes}
        
        # Parameters
        self.level_threshold = level_threshold
        self.level_proximity = level_proximity
        self.max_levels = max_levels
        
    def add_price_point(self, timestamp: datetime, price: float, volume: Optional[float] = None):
        """
        Add a new price point to the dataset and update candles and levels.
        
        Args:
            timestamp (datetime): The timestamp of the price point
            price (float): The price value
            volume (float, optional): The volume at this price point
        """
        if isinstance(timestamp, str):
            try:
                timestamp = dt.datetime.fromisoformat(timestamp)
            except ValueError:
                raise ValueError("Timestamp must be a datetime object")
        
        # Add raw data point
        data_point = {
            'timestamp': timestamp,
            'price': price,
            'volume': volume or 0
        }
        self.raw_data.append(data_point)
        
        # Update candles for each timeframe
        for tf_name, tf_delta in self.timeframes.items():
            self._update_candles(tf_name, tf_delta, data_point)
        
        # Update levels after adding the new data point
        self._update_levels()
        
        # Trim historical data if needed
        self._trim_data()
        
    def _update_candles(self, timeframe, delta, data_point):
        """
        Update the OHLC candles for a specific timeframe.
        
        Args:
            timeframe (str): Name of the timeframe
            delta (timedelta): The timedelta for this timeframe
            data_point (dict): The new price data point
        """
        candles = self.candles[timeframe]
        
        if not candles:
            # Create the first candle
            candle_timestamp = self._normalize_timestamp(data_point['timestamp'], delta)
            candles.append({
                'timestamp': candle_timestamp,
                'open': data_point['price'],
                'high': data_point['price'],
                'low': data_point['price'],
                'close': data_point['price'],
                'volume': data_point['volume']
            })
        else:
            # Check if the new data point belongs to the current candle or needs a new one
            current_candle = candles[-1]
            candle_timestamp = self._normalize_timestamp(data_point['timestamp'], delta)
            
            if candle_timestamp == current_candle['timestamp']:
                # Update current candle
                current_candle['high'] = max(current_candle['high'], data_point['price'])
                current_candle['low'] = min(current_candle['low'], data_point['price'])
                current_candle['close'] = data_point['price']
                current_candle['volume'] += data_point['volume']
            else:
                # Create a new candle
                candles.append({
                    'timestamp': candle_timestamp,
                    'open': data_point['price'],
                    'high': data_point['price'],
                    'low': data_point['price'],
                    'close': data_point['price'],
                    'volume': data_point['volume']
                })
                
    def _normalize_timestamp(self, timestamp, delta):
        """
        Normalize a timestamp to the start of its respective candle period.
        
        Args:
            timestamp (datetime): The timestamp to normalize
            delta (timedelta): The timedelta for the timeframe
            
        Returns:
            datetime: The normalized timestamp
        """
        if delta == timedelta(minutes=1):
            return timestamp.replace(second=0, microsecond=0)
        elif delta == timedelta(minutes=5):
            minutes = (timestamp.minute // 5) * 5
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
        elif delta == timedelta(minutes=15):
            minutes = (timestamp.minute // 15) * 15
            return timestamp.replace(minute=minutes, second=0, microsecond=0)
        elif delta == timedelta(hours=1):
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif delta == timedelta(hours=4):
            hours = (timestamp.hour // 4) * 4
            return timestamp.replace(hour=hours, minute=0, second=0, microsecond=0)
        elif delta == timedelta(days=1):
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            # For custom timeframes, use floor division
            seconds = int(delta.total_seconds())
            timestamp_seconds = int(timestamp.timestamp())
            normalized_seconds = (timestamp_seconds // seconds) * seconds
            return datetime.fromtimestamp(normalized_seconds)
    
    def _update_levels(self):
        """
        Update price levels for all timeframes.
        """
        for tf_name in self.timeframes:
            if len(self.candles[tf_name]) > 5:  # Need at least a few candles
                self._detect_levels(tf_name)
                
    def _detect_levels(self, timeframe):
        """
        Detect price levels for a specific timeframe.
        
        Args:
            timeframe (str): Name of the timeframe
        """
        candles = self.candles[timeframe][-self.lookback_periods[timeframe]:]
        
        if not candles:
            return
        
        # Extract relevant price points (high, low, open, close)
        price_points = []
        for candle in candles:
            price_points.extend([
                candle['high'],
                candle['low']
            ])
            # Optionally include open/close for more precision
            # price_points.extend([candle['open'], candle['close']])
        
        # Create a histogram of price levels
        min_price = min(price_points) * 0.9
        max_price = max(price_points) * 1.1
        
        # Calculate bin size based on level_proximity
        bin_size = (max_price - min_price) * self.level_proximity
        num_bins = int((max_price - min_price) / bin_size) + 1
        
        # Create histogram
        hist, bin_edges = np.histogram(price_points, bins=num_bins, range=(min_price, max_price))
        
        # Find significant levels (bins with counts above threshold)
        threshold_count = len(price_points) * self.level_threshold
        significant_indices = np.where(hist > threshold_count)[0]
        
        # Extract level prices and strengths
        levels = []
        for idx in significant_indices:
            level_price = (bin_edges[idx] + bin_edges[idx + 1]) / 2
            strength = hist[idx] / len(price_points)  # Normalize strength
            
            levels.append({
                'price': level_price,
                'strength': strength,
                'count': hist[idx],
                'timeframe': timeframe,
                'last_touch': self._find_last_touch(level_price, candles)
            })
        
        # Sort levels by strength and keep only the top ones
        levels.sort(key=lambda x: x['strength'], reverse=True)
        self.levels[timeframe] = levels[:self.max_levels]
        
    def _find_last_touch(self, level_price, candles):
        """
        Find the last time a price level was touched.
        
        Args:
            level_price (float): The price level
            candles (list): List of candle dictionaries
            
        Returns:
            datetime: The timestamp of the last touch
        """
        proximity = level_price * self.level_proximity
        
        for candle in reversed(candles):
            # Check if the price level was within the candle's range
            if (candle['low'] - proximity <= level_price <= candle['high'] + proximity):
                return candle['timestamp']
        
        return None
    
    def _trim_data(self):
        """
        Trim historical data to save memory, keeping only what's needed
        for the longest lookback period.
        """
        max_lookback = max(self.lookback_periods.values())
        
        # Trim candles
        for tf in self.timeframes:
            if len(self.candles[tf]) > max_lookback * 2:  # Keep some extra buffer
                self.candles[tf] = self.candles[tf][-max_lookback * 2:]
        
        # Trim raw data - keep approximately enough for the highest timeframe
        max_timeframe = max(self.timeframes.values(), key=lambda x: x.total_seconds())
        max_raw_lookback = int(max_lookback * max_timeframe.total_seconds() / 60) * 2  # Assuming raw data is minute-based
        
        if len(self.raw_data) > max_raw_lookback:
            self.raw_data = self.raw_data[-max_raw_lookback:]
    
    def get_nearest_levels(self, price, n=3, timeframe=None):
        """
        Get the nearest n price levels to a given price.
        
        Args:
            price (float): The reference price
            n (int): Number of levels to return
            timeframe (str, optional): Specific timeframe to check, or None for all timeframes
            
        Returns:
            list: List of the nearest n levels with their details
        """
        all_levels = []
        
        if timeframe:
            # Get levels only from specified timeframe
            all_levels = self.levels.get(timeframe, [])
        else:
            # Combine levels from all timeframes
            for tf_levels in self.levels.values():
                all_levels.extend(tf_levels)
        
        # Calculate distance from current price
        for level in all_levels:
            level['distance'] = abs(level['price'] - price) / price  # Normalized distance
        
        # Sort by distance and return top n
        nearest_levels = sorted(all_levels, key=lambda x: x['distance'])[:n]
        
        return nearest_levels
    
    def get_strongest_levels(self, n=5, timeframe=None):
        """
        Get the strongest n price levels.
        
        Args:
            n (int): Number of levels to return
            timeframe (str, optional): Specific timeframe to check, or None for all timeframes
            
        Returns:
            list: List of the strongest n levels with their details
        """
        all_levels = []
        
        if timeframe:
            # Get levels only from specified timeframe
            all_levels = self.levels.get(timeframe, [])
        else:
            # Combine levels from all timeframes
            for tf_levels in self.levels.values():
                all_levels.extend(tf_levels)
        
        # Sort by strength and return top n
        strongest_levels = sorted(all_levels, key=lambda x: x['strength'], reverse=True)[:n]
        
        return strongest_levels
    
    def is_near_level(self, price, threshold_percent=0.5):
        """
        Check if a price is near any significant level.
        
        Args:
            price (float): The price to check
            threshold_percent (float): How close the price needs to be to a level (as % of price)
            
        Returns:
            dict: The nearest level if within threshold, None otherwise
        """
        nearest = self.get_nearest_levels(price, 1)
        
        if nearest and nearest[0]['distance'] * 100 <= threshold_percent:
            return nearest[0]
        return None
    
    def get_stats(self):
        """
        Get statistics about the tracked data and levels.
        
        Returns:
            dict: Statistics about the data and levels
        """
        stats = {
            'raw_data_points': len(self.raw_data),
            'candles_by_timeframe': {tf: len(candles) for tf, candles in self.candles.items()},
            'levels_by_timeframe': {tf: len(levels) for tf, levels in self.levels.items()},
            'total_levels': sum(len(levels) for levels in self.levels.values()),
            'strongest_level': None,
            'price_range': None
        }
        
        # Find strongest overall level
        all_levels = []
        for tf_levels in self.levels.values():
            all_levels.extend(tf_levels)
            
        if all_levels:
            strongest = max(all_levels, key=lambda x: x['strength'])
            stats['strongest_level'] = {
                'price': strongest['price'],
                'strength': strongest['strength'],
                'timeframe': strongest['timeframe']
            }
        
        # Calculate price range from raw data
        if self.raw_data:
            prices = [point['price'] for point in self.raw_data]
            stats['price_range'] = {
                'min': min(prices),
                'max': max(prices),
                'current': prices[-1]
            }
        
        return stats
