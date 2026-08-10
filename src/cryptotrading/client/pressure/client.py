import os
import logging
from typing import Dict, Any, Optional
import requests

logger = logging.getLogger("cryptotrading.client.pressure")

PRESSURE_SERVICE_URL = os.environ.get("PRESSURE_SERVICE_URL", "http://pressure:8000")

class PressureServiceClient:
    """Client for services/pressure (Pressure Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or PRESSURE_SERVICE_URL).rstrip('/')
        self.timeout = timeout

    def health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        resp = requests.get(url, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()

    def compute_pressure(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}/compute"
        resp = requests.post(url, json=payload, timeout=self.timeout)
        resp.raise_for_status()
        return resp.json()
