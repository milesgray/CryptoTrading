# Active Context: Dynamic Granularity Forecasting & Embed Service Self-Healing

## Quick Reference
- **Feature**: Dynamic Retrieval Forecasting, Frontend Integration, and Embed Service Auto-Training
- **Status**: Completed & Verified ✅

## Executive Summary
Wired up the frontend's segment length (15, 30, 60, or 120 steps) and retrieval frequency (1m, 5m, 15m, 1h) settings to the pattern matching forecasting engine. Designed a thread-safe dynamic in-memory vector index cache on the backend. Additionally, implemented a self-healing auto-population process inside the embed service that automatically trains the CNN contrastive encoder on a limited chronological history, saves the model checkpoint, and regenerates pgvector database setups with real embeddings if weights are missing.

## Architecture Overview
1. **Frontend controls**: Linked `frequency` and `segmentLength` state variables directly to the `/api/retrieval/forecast` URL. Updated React hooks dependencies to trigger queries immediately on settings update.
2. **Dynamic labels & ranges**: Calculates baseline candlestick retrieval date bounds and maps ECharts X-axis indices dynamically based on selected frequency and segment length.
3. **Gateway Router proxy**: Serve router proxy forwards `granularity` and `window_size` parameters with a 30-second client timeout.
4. **Backend Retrieval Cache**: Implemented a thread-safe dictionary cache (`forecasters_cache`) keyed by `(token, granularity, window_size)`. If a combination doesn't exist, it is built on-the-fly, dynamically adjusting STFT frame/hop sizes, FFT bins, and forecast horizon based on target window width.
5. **Embed Service Auto-Training**: Detects missing `encoder.pt` at container startup, truncates setups, trains the model dynamically, saves the checkpoint, generates embeddings, and populates TimescaleDB setups. Uses host volume mounts for persistence.

## Tech Stack for This Feature
- **React + ECharts**: Chart rendering and settings controls
- **FastAPI / HTTPX / asyncio**: Backend service and HTTP proxying
- **numpy / scipy**: Fourier transform (STFT), Pearson correlation, Jensen-Shannon divergence
- **PyTorch**: Contrastive representation learning encoder and DataLoader

## Key Files Modified
- [main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): In-memory index caching, thread-safe dynamic index construction, and expanded `/forecast` endpoint.
- [retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/retrieval.py): Forwarding new params, increasing proxy client timeout.
- [RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx): Link React state variables to request, auto-trigger on change, and format ECharts labels/series positions dynamically.
- [server.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/server.py): Auto-train contrastive encoder, clear database setups, save model weights, and generate/populate pgvector embeddings.
- [trainer.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/models/trainer.py): Fix container import references, and dynamically scale loader batch size/drop_last to prevent ZeroDivisionError.
- [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml): Mount trained weights directory as a persistent volume.

## Verification & Validation
- **Unit Tests**: Executed `pytest tests/test_specretf.py tests/test_forecaster.py` successfully (all 8 tests passed).
- **Vite Build**: Compiled the React production build successfully with no errors.
- **Auto-Training Startup**: Verified through container logs that the `embed` service automatically triggers CPU-bound training, saves model checkpoints, and populates valid embeddings to TimescaleDB. Subsequent restarts successfully load the model from the mounted host volume.
