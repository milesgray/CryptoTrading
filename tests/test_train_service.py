import os
import time
import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

from services.train.main import app, tasks


@pytest.fixture
def dummy_csv(tmp_path):
    """Generates a temporary CSV file with dummy price data for training tests."""
    csv_path = tmp_path / "dummy_prices.csv"
    dates = pd.date_range(start='2025-01-01', periods=100, freq='h')
    prices = np.sin(np.linspace(0, 10, 100)) * 10 + 100 + np.random.normal(0, 0.5, 100)
    df = pd.DataFrame({
        'date': dates,
        'OT': prices
    })
    df.to_csv(csv_path, index=False)
    return str(csv_path)


def test_http_training_lifecycle(dummy_csv, tmp_path):
    # Ensure checkpoints directory is set to a temporary folder
    checkpoints_dir = str(tmp_path / "checkpoints")
    
    client = TestClient(app)
    
    # 1. Start a training task
    train_payload = {
        "task_name": "short_term_forecast",
        "model_id": "test_http_linear",
        "model": "Linear",
        "data": "custom",
        "root_path": os.path.dirname(dummy_csv),
        "data_path": dummy_csv,
        "features": "S",
        "target": "OT",
        "seq_len": 24,
        "label_len": 12,
        "pred_len": 12,
        "enc_in": 1,
        "dec_in": 1,
        "c_out": 1,
        "d_model": 16,
        "train_epochs": 1,
        "batch_size": 4,
        "checkpoints": checkpoints_dir,
        "use_gpu": False,
        "use_tqdm": False
    }
    
    response = client.post("/train", json=train_payload)
    assert response.status_code == 200
    res_data = response.json()
    assert "task_id" in res_data
    assert res_data["status"] == "queued"
    
    task_id = res_data["task_id"]
    
    # 2. Poll for task completion (maximum 10 seconds)
    completed = False
    for _ in range(20):
        time.sleep(0.5)
        status_resp = client.get(f"/tasks/{task_id}")
        assert status_resp.status_code == 200
        task_data = status_resp.json()
        if task_data["status"] == "success":
            completed = True
            break
        elif task_data["status"] == "failed":
            pytest.fail(f"Training task failed: {task_data.get('error')}")
            
    assert completed, "Training task did not complete in time"
    
    # 3. List tasks
    list_tasks_resp = client.get("/tasks")
    assert list_tasks_resp.status_code == 200
    task_list = list_tasks_resp.json()
    assert len(task_list) >= 1
    assert any(t["task_id"] == task_id for t in task_list)
    
    # 4. List models
    list_models_resp = client.get(f"/models?checkpoints_dir={checkpoints_dir}")
    assert list_models_resp.status_code == 200
    model_list = list_models_resp.json()
    assert len(model_list) >= 1
    
    model_info = model_list[0]
    model_id = model_info["model_id"]
    assert model_info["has_config"] is True
    
    # 5. Download model checkpoint
    download_resp = client.get(f"/models/{model_id}/download?checkpoints_dir={checkpoints_dir}")
    assert download_resp.status_code == 200
    assert len(download_resp.content) > 0
    
    # 6. Run model inference (predict endpoint)
    # The Linear model expects input shape [batch_size, seq_len, enc_in] -> [1, 24, 1]
    dummy_input = {
        "x": [[[100.0 + i] for i in range(24)]]
    }
    predict_resp = client.post(f"/models/{model_id}/predict?checkpoints_dir={checkpoints_dir}", json=dummy_input)
    assert predict_resp.status_code == 200
    pred_data = predict_resp.json()
    assert "predictions" in pred_data
    predictions = pred_data["predictions"]
    # Output should have shape [batch_size, pred_len, c_out] -> [1, 12, 1]
    assert len(predictions) == 1
    assert len(predictions[0]) == 12
    assert len(predictions[0][0]) == 1
