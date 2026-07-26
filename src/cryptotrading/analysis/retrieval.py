import numpy as np
from scipy.signal import welch
from scipy.fft import fft
from typing import Tuple, Dict, Any, Optional
import datetime as dt

from cryptotrading.analysis.book import OrderBookFeaturizer, OrderBookSnapshot
from cryptotrading.analysis.levels import PriceLevels

class RetrievalEncoder:
    def __init__(
        self,
        forecast_size: int = 15,
        historic_window_size: Optional[int] = None,
        n_fft: int = 32
    ):
        self.forecast_size = forecast_size
        self.historic_window_size = historic_window_size if historic_window_size is not None else 4 * forecast_size
        self.n_fft = n_fft
        self.orderbook_featurizer = OrderBookFeaturizer()
        self.price_levels_detector = PriceLevels()

    def extract_spectral_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract Fourier/Welch spectral features."""
        if len(prices) < self.historic_window_size:
            prices = np.pad(prices, (0, self.historic_window_size - len(prices)), 'constant')
        fft_coeffs = np.abs(fft(prices, n=self.n_fft))
        _, psd = welch(prices, nperseg=self.n_fft)
        return np.concatenate([fft_coeffs, psd])

    def extract_orderbook_features(self, order_book: dict, current_price: float = 1.0, token: str = "BTC") -> np.ndarray:
        """Extract comprehensive order book features using OrderBookFeaturizer."""
        bids = order_book.get("bids", [])
        asks = order_book.get("asks", [])
        
        # Convert tuples/lists if necessary
        formatted_bids = [(float(p), float(v)) for p, v in bids] if bids else []
        formatted_asks = [(float(p), float(v)) for p, v in asks] if asks else []

        if not formatted_bids or not formatted_asks:
            snapshot = OrderBookSnapshot(
                timestamp=dt.datetime.now(dt.timezone.utc).timestamp(),
                bids=[(current_price * 0.999, 1.0)],
                asks=[(current_price * 1.001, 1.0)],
                mid_price=current_price
            )
        else:
            best_bid = formatted_bids[0][0]
            best_ask = formatted_asks[0][0]
            mid_price = (best_bid + best_ask) / 2.0 if best_ask > best_bid else current_price
            snapshot = OrderBookSnapshot(
                timestamp=dt.datetime.now(dt.timezone.utc).timestamp(),
                bids=formatted_bids,
                asks=formatted_asks,
                mid_price=mid_price
            )
        
        features_dict = self.orderbook_featurizer.extract_features(snapshot, token=token, validate=False)
        return self.orderbook_featurizer.flatten_features(features_dict)

    def extract_price_level_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract support/resistance price level features using PriceLevels."""
        detector = PriceLevels()
        base_time = dt.datetime.now(dt.timezone.utc) - dt.timedelta(minutes=len(prices))
        for idx, price in enumerate(prices):
            timestamp = base_time + dt.timedelta(minutes=idx)
            detector.add_price_point(timestamp, float(price))
            
        last_price = float(prices[-1]) if len(prices) > 0 else 1.0
        
        # Collect all detected levels
        all_levels = []
        for tf_levels in detector.levels.values():
            all_levels.extend(tf_levels)
            
        if not all_levels:
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            
        # Calculate normalized distance manually/safely
        for l in all_levels:
            l.distance = abs(l.level - last_price) / (last_price + 1e-8)
            
        nearest = sorted(all_levels, key=lambda x: getattr(x, 'distance', 0.0))[:2]
        strongest = sorted(all_levels, key=lambda x: x.strength, reverse=True)[:2]
        
        near_dist1 = nearest[0].distance if len(nearest) > 0 else 0.0
        near_dist2 = nearest[1].distance if len(nearest) > 1 else 0.0
        strong1 = strongest[0].strength if len(strongest) > 0 else 0.0
        strong2 = strongest[1].strength if len(strongest) > 1 else 0.0
        
        total_levels = float(len(all_levels))
        
        return np.array([near_dist1, near_dist2, strong1, strong2, total_levels], dtype=np.float32)

    def encode(self, prices: np.ndarray, order_book: dict, token: str = "BTC") -> np.ndarray:
        """Encode a price segment + order book into a comprehensive feature vector."""
        spectral = self.extract_spectral_features(prices)
        current_price = float(prices[-1]) if len(prices) > 0 else 1.0
        orderbook = self.extract_orderbook_features(order_book, current_price=current_price, token=token)
        price_levels = self.extract_price_level_features(prices)
        
        timeseries = np.array([
            prices[-1] if len(prices) > 0 else 0.0,
            np.mean(prices) if len(prices) > 0 else 0.0,
            np.std(prices) if len(prices) > 0 else 0.0,
            (prices[-1] - prices[0]) / (prices[0] + 1e-8) if len(prices) > 0 else 0.0
        ], dtype=np.float32)
        
        return np.concatenate([spectral, orderbook, price_levels, timeseries])