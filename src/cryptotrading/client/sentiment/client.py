import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.sentiment")

SENTIMENT_SERVICE_URL = os.environ.get("SENTIMENT_SERVICE_URL", "http://sentiment:8000")

class SentimentServiceClient:
    """Client for services/sentiment (Sentiment Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or SENTIMENT_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def analyze(self, text: str) -> Dict[str, Any]:
        url = f"{self.base_url}/analyze"
        resp = requests.post(url, json={"text": text}, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
