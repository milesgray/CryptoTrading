import os
import time
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

    def is_healthy(self) -> bool:
        try:
            res = self.health()
            return res.get("status") in ["ok", "healthy", "success", "running"] or bool(res)
        except Exception:
            return False

    def check_health(self) -> Dict[str, Any]:
        start = time.time()
        try:
            data = self.health()
            latency_ms = round((time.time() - start) * 1000, 2)
            healthy = data.get("status") in ["ok", "healthy", "success", "running"] or bool(data)
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
