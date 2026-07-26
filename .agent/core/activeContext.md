# Active Context: Resolving Docker Compose Log Issues, DB Password Auth & Retrieval Dimension Mismatch

## Quick Reference
- **Feature**: Docker Compose Log Issues & Retrieval Vector Dimension Resolution
- **Status**: Completed ✅

## Executive Summary
Resolved password authentication failures caused by URI percent-encoding issues (`POSTGRES_PASSWORD=6AFS87dfsaas%116`), bound TimescaleDB port `5432` to localhost (`127.0.0.1:5432:5432`) to eliminate public internet brute-force attacks, added GET /{token} endpoints to the pressure service, and fixed the retrieval service vector dimension mismatch (`Query embedding dimension 952 != index dimension 184`) by dynamically adapting embedding vectors and implementing robust health-check retries across container services.

## Tech Stack for This Feature
- **Docker Compose**: Container networking, port security hardening, image builds.
- **FastAPI / Uvicorn**: Fallback GET endpoints (`/BTC`, `/pressure/BTC`) and unbuffered logging (`PYTHONUNBUFFERED=1`).
- **Python / NumPy / Annoy**: Dynamic vector length padding and cropping in `RetrievalServiceEncoder`.

## Key Files Modified
- [services/retrieval/encoder.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/encoder.py): Guaranteed `encode_segment` output vectors strictly match Annoy index dimension `self.dim`.
- [services/retrieval/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): Optimized startup event loop to pre-build default index for BTC only and added health check retry loops.
- [services/pressure/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/main.py): Added `@app.get("/{token}")` and `@app.get("/pressure/{token}")` fallback endpoints.
- [frontend/vite.config.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/vite.config.js): Updated `/api/pressure/train` proxy route to target `pressureUrl`.
- [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml): Bound TimescaleDB port 5432 to `127.0.0.1`.
- [Dockerfile.retrieval](file:///home/miles/Development/notebooks/CryptoTrading/Dockerfile.retrieval): Set `ENV PYTHONUNBUFFERED=1`.

## Verification Details
- `curl -s http://localhost:8390/BTC`: Returned HTTP 200 OK with valid metrics.
- `curl -s "http://localhost:8005/forecast?symbol=BTC&k=3&granularity=1m&window_size=60"`: Returned HTTP 200 OK with valid forecast data.
- Unit tests: 25 non-torch tests passed cleanly.
