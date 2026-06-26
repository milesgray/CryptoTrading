import asyncio
import logging
from typing import Dict
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from cryptotrading.util.orchestrator import ServiceOrchestrator

logger = logging.getLogger("fastapi_server")

router = APIRouter(prefix="/services")
ws_router = APIRouter(prefix="/ws/services")

orchestrator = ServiceOrchestrator()

class ConfigUpdatePayload(BaseModel):
    config: Dict[str, str]

@router.get("")
async def get_services():
    return orchestrator.get_services_status()

@router.post("/{name}/start")
async def start_service_endpoint(name: str):
    result = orchestrator.start_service(name)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

@router.post("/{name}/stop")
async def stop_service_endpoint(name: str):
    result = orchestrator.stop_service(name)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

@router.post("/{name}/restart")
async def restart_service_endpoint(name: str):
    result = orchestrator.restart_service(name)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

@router.get("/{name}/logs")
async def get_service_logs_endpoint(name: str, limit: int = 100):
    logs = orchestrator.get_service_logs(name, limit)
    return {"logs": logs}

@router.post("/{name}/config")
async def update_service_config_endpoint(name: str, payload: ConfigUpdatePayload):
    result = orchestrator.update_service_config(name, payload.config)
    if result.get("status") == "error":
        raise HTTPException(status_code=400, detail=result.get("message"))
    return result

@ws_router.websocket("/status")
async def websocket_services_status(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status_data = orchestrator.get_services_status()
            await websocket.send_json({
                "type": "services_status",
                "data": status_data
            })
            await asyncio.sleep(1.5)
    except WebSocketDisconnect:
        logger.debug("Services status WebSocket disconnected")
    except Exception as e:
        logger.error(f"Services status WebSocket error: {e}")
