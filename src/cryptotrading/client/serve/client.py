import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.serve")

SERVE_SERVICE_URL = os.environ.get("SERVE_SERVICE_URL", "http://serve:8000")

class ServeServiceClient:
    """Client for services/serve (API Serving Server)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or SERVE_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_latest_price(self, token: str) -> Dict[str, Any]:
        url = f"{self.base_url}/latest_price/{token}"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_candlestick(self, token: str, granularity: str = "1m", limit: int = 100) -> Dict[str, Any]:
        url = f"{self.base_url}/candlestick/{token}"
        params = {"granularity": granularity, "limit": limit}
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
