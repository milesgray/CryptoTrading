import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.trade")

TRADE_SERVICE_URL = os.environ.get("TRADE_SERVICE_URL", "http://trade:8000")

class TradeServiceClient:
    """Client for services/trade (Trade Execution Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or TRADE_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def execute_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/order"
        resp = requests.post(url, json=order, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
