import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.predict")

PREDICT_SERVICE_URL = os.environ.get("PREDICT_SERVICE_URL", "http://predict:8000")

class PredictServiceClient:
    """Client for services/predict (Prediction Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or PREDICT_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/predict"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
