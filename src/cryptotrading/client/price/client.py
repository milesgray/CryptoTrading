import os
import time
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("cryptotrading.client.price")

PRICE_SERVICE_URL = os.environ.get("PRICE_SERVICE_URL", "http://price:8387")

class PriceServiceClient:
    """Client for services/price (Price Ingestion & Aggregation Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or PRICE_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/health"
            resp = requests.get(url, timeout=self.timeout)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
        # Fallback to /status if /health is not available
        url = f"{self.base_url}/status"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def is_healthy(self) -> bool:
        try:
            res = self.health()
            return res.get("status") in ["ok", "healthy", "success", "running"] or res.get("running") is True or bool(res)
        except Exception:
            return False

    def check_health(self) -> Dict[str, Any]:
        start = time.time()
        try:
            data = self.health()
            latency_ms = round((time.time() - start) * 1000, 2)
            healthy = data.get("status") in ["ok", "healthy", "success", "running"] or data.get("running") is True or bool(data)
            return {
                "healthy": healthy,
                "status": data.get("status", "healthy") if healthy else "degraded",
                "latency_ms": latency_ms,
                "details": data,
                "error": None
            }
        except Exception as e:
            latency_ms = round((time.time() - start) * 1000, 2)
            return {
                "healthy": False,
                "status": "unhealthy",
                "latency_ms": latency_ms,
                "details": {},
                "error": str(e)
            }

    def get_status(self) -> Dict[str, Any]:
        url = f"{self.base_url}/status"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_price_system_status(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        url = f"{self.base_url}/price_system_status"
        params = {"symbol": symbol} if symbol else {}
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def get_logs(self, limit: int = 100, level: Optional[str] = None) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/logs"
        params = {"limit": limit}
        if level:
            params["level"] = level
        resp = requests.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
