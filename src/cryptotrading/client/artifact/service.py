import os
import time
import logging
from typing import Optional, List, Dict, Any
import requests

logger = logging.getLogger("cryptotrading.artifact_client")

ARTIFACT_SERVICE_URL = os.environ.get("ARTIFACT_SERVICE_URL", "http://artifact:8383")


class ArtifactServiceClient:
    """Client wrapper for services/artifact (Model and Checkpoint Artifact Service)"""
    def __init__(self, base_url: Optional[str] = None, timeout: float = 30.0):
        self.base_url = (base_url or ARTIFACT_SERVICE_URL).rstrip('/')
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

    def download_artifact(self, category: str, filename: str, local_destination_path: str) -> bool:
        url = f"{self.base_url}/download/{category}/{filename}"
        try:
            response = requests.get(url, stream=True, timeout=self.timeout)
            if response.status_code == 200:
                os.makedirs(os.path.dirname(local_destination_path), exist_ok=True)
                with open(local_destination_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                logger.info(f"Downloaded artifact {category}/{filename} -> {local_destination_path}")
                return True
            else:
                logger.warning(f"Artifact {category}/{filename} not found on server (HTTP {response.status_code})")
                return False
        except Exception as e:
            logger.error(f"Failed to download artifact {category}/{filename}: {e}")
            return False

    def upload_artifact(self, local_file_path: str, category: str, filename: Optional[str] = None, overwrite: bool = True) -> bool:
        if not os.path.exists(local_file_path):
            logger.error(f"Cannot upload artifact: Local file '{local_file_path}' does not exist.")
            return False

        upload_filename = filename or os.path.basename(local_file_path)
        url = f"{self.base_url}/upload"
        
        try:
            with open(local_file_path, "rb") as f:
                files = {"file": (upload_filename, f, "application/octet-stream")}
                data = {"category": category, "overwrite": str(overwrite).lower()}
                response = requests.post(url, files=files, data=data, timeout=max(self.timeout, 60.0))
                
            if response.status_code == 200:
                logger.info(f"Uploaded artifact {local_file_path} -> {category}/{upload_filename}")
                return True
            else:
                logger.error(f"Failed to upload artifact: HTTP {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Failed to upload artifact {local_file_path}: {e}")
            return False

    def list_category_files(self, category: str) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/files/{category}"
        try:
            response = requests.get(url, timeout=min(self.timeout, 10.0))
            if response.status_code == 200:
                return response.json().get("files", [])
            return []
        except Exception as e:
            logger.error(f"Failed to list category files for '{category}': {e}")
            return []


# Default singleton instance for convenience
_default_artifact_client = ArtifactServiceClient()

def download_artifact(category: str, filename: str, local_destination_path: str) -> bool:
    return _default_artifact_client.download_artifact(category, filename, local_destination_path)

def upload_artifact(local_file_path: str, category: str, filename: Optional[str] = None, overwrite: bool = True) -> bool:
    return _default_artifact_client.upload_artifact(local_file_path, category, filename, overwrite)

def list_category_files(category: str) -> List[Dict[str, Any]]:
    return _default_artifact_client.list_category_files(category)
