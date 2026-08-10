import os
import logging
from typing import Dict, Any, Optional, List
import requests

logger = logging.getLogger("cryptotrading.client.retrieval")

RETRIEVAL_SERVICE_URL = os.environ.get("RETRIEVAL_SERVICE_URL", "http://retrieval:8000")

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
