import os
import pytest
import tempfile
from fastapi.testclient import TestClient

from services.artifact.main import app, validate_safe_path, DEFAULT_STORAGE_DIR

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"


def test_upload_and_download_artifact():
    test_content = b"dummy model weight data 12345"
    file_tuple = ("test_model.bin", test_content, "application/octet-stream")
    
    # 1. Upload
    upload_resp = client.post(
        "/upload",
        files={"file": file_tuple},
        data={"category": "test_cat", "overwrite": "true"}
    )
    assert upload_resp.status_code == 200
    upload_data = upload_resp.json()
    assert upload_data["success"] is True
    assert upload_data["filename"] == "test_model.bin"
    assert upload_data["category"] == "test_cat"
    
    # 2. Download
    download_resp = client.get("/download/test_cat/test_model.bin")
    assert download_resp.status_code == 200
    assert download_resp.content == test_content
    
    # 3. List Category Files
    list_resp = client.get("/files/test_cat")
    assert list_resp.status_code == 200
    files = list_resp.json().get("files", [])
    assert any(f["filename"] == "test_model.bin" for f in files)
    
    # 4. Delete Artifact
    del_resp = client.delete("/files/test_cat/test_model.bin")
    assert del_resp.status_code == 200
    assert del_resp.json()["success"] is True


def test_path_traversal_prevention():
    with pytest.raises(Exception):
        validate_safe_path("../../etc", "passwd")
