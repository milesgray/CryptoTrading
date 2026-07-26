import unittest
from unittest.mock import AsyncMock, patch
# pyrefly: ignore [missing-import]
from fastapi.testclient import TestClient

# pyrefly: ignore [missing-import]
from main import app
# pyrefly: ignore [missing-import]
from cryptotrading.analysis.book import OrderBookSnapshot

class TestRealtimeTokenPressure(unittest.TestCase):
    def setUp(self):
        # pyrefly: ignore [missing-import]
        import main
        if main.featurizer is None:
            # pyrefly: ignore [missing-import]
            from cryptotrading.analysis.book import OrderBookFeaturizer
            main.featurizer = OrderBookFeaturizer()
        self.client = TestClient(app)

    @patch("data_loader.OrderBookDataLoader.initialize", new_callable=AsyncMock)
    @patch("data_loader.OrderBookDataLoader.load_orderbook_data", new_callable=AsyncMock)
    def test_get_token_pressure_realtime(self, mock_load_data, mock_init):
        # Mock orderbook snapshots returned from database
        now_ts = 1700000000.0
        snapshot = OrderBookSnapshot(
            timestamp=now_ts,
            bids=[(60000.0, 5.0), (59990.0, 10.0)],
            asks=[(60010.0, 2.0), (60020.0, 4.0)],
            mid_price=60005.0
        )
        mock_load_data.return_value = ([snapshot], None)

        response = self.client.get("/pressure/BTC")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Ensure values are calculated dynamically, not static dummy defaults
        self.assertIn("buy_pressure", data)
        self.assertIn("sell_pressure", data)
        self.assertIn("total_pressure", data)
        self.assertIn("ofi", data)
        self.assertIn("cvd", data)
        self.assertIn("bap", data)
        self.assertNotEqual(data["cvd"], 2500)  # Previously static dummy value 2500
        self.assertNotEqual(data["bap"], 50.0)  # Previously static dummy value 50.0

    @patch("data_loader.OrderBookDataLoader.initialize", new_callable=AsyncMock)
    @patch("data_loader.OrderBookDataLoader.load_orderbook_data", new_callable=AsyncMock)
    def test_get_token_pressure_no_snapshots_returns_404(self, mock_load_data, mock_init):
        # Empty DB return triggers HTTP 404 error instead of fake fallback data
        mock_load_data.return_value = ([], None)

        response = self.client.get("/BTC")
        self.assertEqual(response.status_code, 404)
        data = response.json()
        self.assertIn("detail", data)
        self.assertIn("No recent orderbook data available", data["detail"])

if __name__ == "__main__":
    unittest.main()
