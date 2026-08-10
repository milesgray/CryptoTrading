import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.train")

TRAIN_SERVICE_URL = os.environ.get("TRAIN_SERVICE_URL", "http://train:8000")

class TrainServiceClient:
    """Client for services/train (Forecasting Model Trainer Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or TRAIN_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def trigger_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/train"
        resp = requests.post(url, json=config, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
