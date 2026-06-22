import numpy as np
from typing import Dict, Any, List

class MockRollbitClient:
    """Mock Rollbit client for testing."""
    def __init__(self):
        pass

    def fetch_order_book(self, symbol: str, limit: int = 20):
        """Mock order book fetch."""
        return {
            "bids": [[100, 10], [99, 5]],
            "asks": [[101, 8], [102, 12]]
        }

class RollbitDataLoader:
    def __init__(self, symbol: str = "BTC"):
        self.client = MockRollbitClient()
        self.symbol = symbol

    def fetch_historical_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Fetch historical price and order book data from Rollbit."""
        data = []
        for i in range(limit):
            data.append({
                "prices": np.random.rand(60) + i * 0.1,
                "order_book": {
                    "bids": [[100 + i, 10]],
                    "asks": [[101 + i, 10]]
                }
            })
        return data

    def fetch_latest_data(self) -> Dict[str, Any]:
        """Fetch latest price and order book data from Rollbit."""
        return {
            "prices": np.random.rand(60) + 50 * 0.1,
            "order_book": {
                "bids": [[150, 10]],
                "asks": [[151, 10]]
            }
        }