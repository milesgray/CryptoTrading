import os
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("cryptotrading.client.embed")

EMBED_SERVICE_URL = os.environ.get("EMBED_SERVICE_URL", "http://embed:8000")

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
