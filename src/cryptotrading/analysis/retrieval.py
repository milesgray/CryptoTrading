import numpy as np
from scipy.signal import welch
from scipy.fft import fft
from typing import Tuple

class RetrievalEncoder:
    def __init__(self, window_size: int = 60, n_fft: int = 32):
        self.window_size = window_size
        self.n_fft = n_fft

    def extract_spectral_features(self, prices: np.ndarray) -> np.ndarray:
        """Extract Fourier/Welch spectral features."""
        if len(prices) < self.window_size:
            prices = np.pad(prices, (0, self.window_size - len(prices)), 'constant')
        fft_coeffs = np.abs(fft(prices, n=self.n_fft))
        _, psd = welch(prices, nperseg=self.n_fft)
        return np.concatenate([fft_coeffs, psd])

    def extract_orderbook_features(self, order_book: dict) -> np.ndarray:
        """Extract order book imbalance, spread, depth."""
        bids = np.array(order_book.get("bids", []))
        asks = np.array(order_book.get("asks", []))
        
        # Handle empty order book
        if len(bids) == 0 or len(asks) == 0:
            return np.array([0.0, 0.0, 0.0])
        
        imbalance = (bids[:, 1].sum() - asks[:, 1].sum()) / (bids[:, 1].sum() + asks[:, 1].sum())
        spread = asks[:, 0].min() - bids[:, 0].max()
        depth = bids[:, 1].sum() + asks[:, 1].sum()
        return np.array([imbalance, spread, depth])

    def encode(self, prices: np.ndarray, order_book: dict) -> np.ndarray:
        """Encode a price segment + order book into a vector."""
        spectral = self.extract_spectral_features(prices)
        orderbook = self.extract_orderbook_features(order_book)
        timeseries = np.array([
            prices[-1],  # Last price
            np.mean(prices),  # Mean
            np.std(prices),  # Volatility
            (prices[-1] - prices[0]) / (prices[0] + 1e-8)  # Momentum (avoid div0)
        ])
        return np.concatenate([spectral, orderbook, timeseries])