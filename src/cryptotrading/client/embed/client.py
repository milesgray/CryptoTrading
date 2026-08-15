import os
import time
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("cryptotrading.client.embed")

EMBED_SERVICE_URL = os.environ.get("EMBED_SERVICE_URL", "http://embed:8380")

class EmbedServiceClient:
    """Client for services/embed (Contrastive Pattern Matching / Embedding Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or EMBED_SERVICE_URL).rstrip('/')
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

    def generate_embedding(self, returns_window: List[float]) -> Dict[str, Any]:
        url = f"{self.base_url}/embed"
        resp = requests.post(url, json={"returns": returns_window}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def generate_batch_embedding(self, price_windows: List[List[float]]) -> Dict[str, Any]:
        url = f"{self.base_url}/embed/batch"
        resp = requests.post(url, json={"price_windows": price_windows}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def search_matches(self, returns_window: List[float], top_k: int = 5) -> Dict[str, Any]:
        url = f"{self.base_url}/search"
        payload = {"returns": returns_window, "top_k": top_k}
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def add_setup(self, **kwargs) -> Dict[str, Any]:
        url = f"{self.base_url}/setup/add"
        resp = requests.post(url, json=kwargs, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
