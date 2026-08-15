import os
import time
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("cryptotrading.client.retrieval")

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval:8388")

class RetrievalServiceClient:
    """Client for services/retrieval (Retrieval Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or RETRIEVAL_SERVICE_URL).rstrip('/')
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

    def query_similar(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/query"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def index_vectors(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/index"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def rebuild(self, symbol: str = "BTC") -> Dict[str, Any]:
        url = f"{self.base_url}/rebuild?symbol={symbol}"
        resp = requests.post(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
