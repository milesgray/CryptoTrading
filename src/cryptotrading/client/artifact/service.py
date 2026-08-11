import os
import logging
from typing import Optional, List, Dict, Any
import requests

logger = logging.getLogger("cryptotrading.artifact_client")

ARTIFACT_SERVICE_URL = os.environ.get("ARTIFACT_SERVICE_URL", "http://artifact:8000")


def download_artifact(category: str, filename: str, local_destination_path: str) -> bool:
    """
    Downloads an artifact from the Artifact Service to a local path.
    If the Artifact Service is unavailable or file is missing, returns False.
    """
    url = f"{ARTIFACT_SERVICE_URL.rstrip('/')}/download/{category}/{filename}"
    try:
        response = requests.get(url, stream=True, timeout=30)
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


def upload_artifact(local_file_path: str, category: str, filename: Optional[str] = None, overwrite: bool = True) -> bool:
    """
    Uploads a local file/artifact to the Artifact Service under the specified category.
    """
    if not os.path.exists(local_file_path):
        logger.error(f"Cannot upload artifact: Local file '{local_file_path}' does not exist.")
        return False

    upload_filename = filename or os.path.basename(local_file_path)
    url = f"{ARTIFACT_SERVICE_URL.rstrip('/')}/upload"
    
    try:
        with open(local_file_path, "rb") as f:
            files = {"file": (upload_filename, f, "application/octet-stream")}
            data = {"category": category, "overwrite": str(overwrite).lower()}
            response = requests.post(url, files=files, data=data, timeout=60)
            
        if response.status_code == 200:
            logger.info(f"Uploaded artifact {local_file_path} -> {category}/{upload_filename}")
            return True
        else:
            logger.error(f"Failed to upload artifact: HTTP {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Failed to upload artifact {local_file_path}: {e}")
        return False


def list_category_files(category: str) -> List[Dict[str, Any]]:
    """
    Lists files in a given category from the Artifact Service.
    """
    url = f"{ARTIFACT_SERVICE_URL.rstrip('/')}/files/{category}"
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            return response.json().get("files", [])
        return []
    except Exception as e:
        logger.error(f"Failed to list category files for '{category}': {e}")
        return []
