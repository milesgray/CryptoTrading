import os
import sys
import time
import pytest
from fastapi.testclient import TestClient

# Add project root and src to path
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cryptotrading.util.orchestrator import ServiceOrchestrator
from services.serve.app import app

@pytest.fixture
def orchestrator():
    """Fixture to get a clean ServiceOrchestrator instance."""
    # ServiceOrchestrator is a singleton, so we just retrieve it
    orch = ServiceOrchestrator()
    # Make sure mock trade service script exists
    orch._create_mock_trade_service()
    return orch

def test_orchestrator_initialization(orchestrator):
    """Test that all 10 microservices are correctly registered."""
    status_list = orchestrator.get_services_status()
    assert len(status_list) == 10
    
    # Assert that all required services are in the status list
    service_names = [s["name"] for s in status_list]
    expected_names = [
        "serve", "price", "retrieval", "embed", "sentiment", 
        "jepa", "pressure", "predict", "train", "trade"
    ]
    for name in expected_names:
        assert name in service_names

    # Assert that 'serve' is marked as RUNNING by default
    serve_status = next(s for s in status_list if s["name"] == "serve")
    assert serve_status["status"] == "RUNNING"
    assert serve_status["pid"] == os.getpid()

def test_orchestrator_lifecycle(orchestrator):
    """Test the full lifecycle (start, monitor, read logs, stop) of a microservice."""
    # Ensure the trade service is stopped initially
    orchestrator.stop_service("trade")
    
    # Start the trade service
    start_res = orchestrator.start_service("trade")
    assert start_res["status"] == "success"
    assert start_res["pid"] is not None
    
    # Give the process a moment to start and write logs
    time.sleep(1.5)
    
    # Check status and resources
    status_list = orchestrator.get_services_status()
    trade_status = next(s for s in status_list if s["name"] == "trade")
    assert trade_status["status"] == "RUNNING"
    assert trade_status["pid"] == start_res["pid"]
    assert trade_status["uptime_seconds"] > 0
    
    # We should have some resource usage (CPU or memory)
    # CPU might be 0.0 if idle, but memory should be > 0.0 MB
    assert trade_status["memory_mb"] > 0.0
    
    # Read logs
    logs = orchestrator.get_service_logs("trade", limit=10)
    assert len(logs) > 0
    assert any("Mock Polymarket Trade Broker started" in line for line in logs)
    
    # Stop the service
    stop_res = orchestrator.stop_service("trade")
    assert stop_res["status"] == "success"
    
    # Give it a split second to update status
    time.sleep(0.5)
    
    # Check that it is stopped
    status_list = orchestrator.get_services_status()
    trade_status = next(s for s in status_list if s["name"] == "trade")
    assert trade_status["status"] == "STOPPED"
    assert trade_status["pid"] is None

def test_serve_endpoints():
    """Test the FastAPI service management endpoints using TestClient."""
    client = TestClient(app)
    
    # 1. Test GET /services
    response = client.get("/services")
    assert response.status_code == 200
    services = response.json()
    assert len(services) == 10
    
    # 2. Test POST /services/trade/start
    # Make sure it's stopped first
    client.post("/services/trade/stop")
    time.sleep(0.5)
    
    start_response = client.post("/services/trade/start")
    assert start_response.status_code == 200
    assert start_response.json()["status"] == "success"
    
    # 3. Test GET /services/trade/logs
    time.sleep(1.0)
    logs_response = client.get("/services/trade/logs?limit=5")
    assert logs_response.status_code == 200
    assert "logs" in logs_response.json()
    assert len(logs_response.json()["logs"]) > 0
    
    # 4. Test POST /services/trade/config
    config_response = client.post("/services/trade/config", json={
        "config": {"TEST_VAR": "test_value_123"}
    })
    assert config_response.status_code == 200
    assert config_response.json()["status"] == "success"
    assert config_response.json()["current_config"]["TEST_VAR"] == "test_value_123"
    
    # 5. Test POST /services/trade/stop
    stop_response = client.post("/services/trade/stop")
    assert stop_response.status_code == 200
    assert stop_response.json()["status"] == "success"
