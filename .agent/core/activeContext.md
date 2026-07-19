# Active Context: FastAPI Pressure Service & Frontend Integration

## Quick Reference
- **Feature**: FastAPI Pressure Service & Frontend Integration
- **Plan File**: N/A (Direct execution of wrap/endpoint request)
- **Status**: Completed ✅

## Executive Summary
Wrapped the PyTorch order book pressure service in a FastAPI server (`services/pressure/main.py`) to expose both model predictions and feature extraction. Implemented background training via `POST /train` and progress status updates via `GET /train/status`. Resolved python packaging/relative import errors in the `pressure` submodules to allow top-level module resolution and clean test execution. Updated Vite and Docker Compose environments to proxy and expose the new service, and integrated a stylized training console inside the frontend's `OrderBookPressurePanel` to launch jobs and visualize progress metrics.

## Tech Stack for This Feature
- **FastAPI / Uvicorn**: Lightweight REST application server and asynchronous background tasks.
- **PyTorch**: Model loading, forward inference, and training checkpoints (`best_model.pt`).
- **React (Vite) / ECharts**: Frontend console and progress dashboard.
- **Docker Compose**: Container networking and proxy definition.

## Key Files Modified
- [services/pressure/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/main.py): [NEW] FastAPI application containing `/features`, `/predict`, `/train`, and `/train/status` endpoints.
- [services/pressure/Dockerfile](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/Dockerfile): Updated CMD to start the FastAPI server with Uvicorn.
- [services/pressure/data_loader.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/data_loader.py): Fixed relative import of `pressure_features` and wrapped `inspect.signature` in a try-except to support mocked adapters in tests.
- [services/pressure/train.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/train.py): Fixed relative model/oracle imports and added `progress_callback` hook to `PressureTrainer.train`.
- [services/pressure/oracle.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/oracle.py): Fixed relative import of `pressure_features`.
- [services/pressure/example_data_loading.py](file:///home/miles/Development/notebooks/CryptoTrading/services/pressure/example_data_loading.py): Fixed relative imports.
- [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml): Declared the `pressure` service on port `8390` and defined `VITE_PRESSURE_URL` on the `frontend` service.
- [frontend/vite.config.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/vite.config.js): Added `/api/pressure` proxying.
- [frontend/src/services/api.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/services/api.js): Added `startPressureTraining` and `getPressureTrainingStatus` Axios callers.
- [frontend/src/components/SpecializedServicePanels.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/SpecializedServicePanels.jsx): Added a reactive training panel with progress bar and loss tracker directly inside the Order Book Pressure dashboard.

## Critical Implementation Details
1. **Absolute Imports Fix**: Shifting from relative (`from .module`) to absolute (`from module`) imports resolved the Python package boundary issue, enabling clean pytest execution when invoking tests directly inside `/home/miles/Development/notebooks/CryptoTrading/services/pressure` and preventing startup crashes inside the `/app` container root directory.
2. **Mock-Safe Signature Resolution**: Wrapping `inspect.signature` in `data_loader.py` prevents tests from failing when resolving mock specs on Python 3.10, reverting to standard default query behaviors.

## Next Steps
- Verify Docker container rebuild of the `pressure` and `frontend` images.
- Initiate test training runs inside the new frontend console interface.
