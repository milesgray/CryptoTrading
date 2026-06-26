import logging
import os
import httpx
from fastapi import APIRouter, HTTPException

logger = logging.getLogger("fastapi_server")

router = APIRouter(prefix="/retrieval")

@router.get("/forecast")
async def forecast(symbol: str = "BTC", k: int = 5):
    """Proxy to retrieval service with robust timeout and error handling."""
    retrieval_url = os.getenv("RETRIEVAL_SERVICE_URL", "http://localhost:8000")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{retrieval_url}/forecast?symbol={symbol}&k={k}")
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as e:
        logger.error(f"Error proxying to retrieval service: {e}")
        raise HTTPException(
            status_code=502,
            detail=f"Retrieval service error: {str(e)}"
        )
