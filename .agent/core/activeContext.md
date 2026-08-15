# Active Context: Retrieval Forecast Method Routing & Real-time Pressure WebSocket

## Quick Reference
- **Feature**: Forecast Method Routing & Real-time Order Book Pressure WebSocket
- **Status**: Completed ✅

## Executive Summary
Fixed the frontend double-chart rendering bug, changed default chart granularity to 1m, wired the forecasting method dropdown in the frontend through the serve service proxy to the retrieval service, and converted frontend order book pressure updates to stream in real-time over WebSockets via `/ws/pressure/{token}`.

## Tech Stack & Components
- **Frontend (React / Vite / ECharts / AnyStock)**: Real-time subscriptions for price, order book, and pressure updates in `WebSocketService`.
- **FastAPI / Uvicorn (`services/serve`)**: Added `/ws/pressure/{token}` endpoint streaming pressure features, and forwarded `method` query parameter in `/api/retrieval/forecast`.
- **Analysis (`cryptotrading.analysis.retrieval`)**: Updated `RetrievalEncoder` constructor to support `window_factor` with backwards compatibility.

## Key Files Modified
- [frontend/src/components/CandlestickChart.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/CandlestickChart.jsx): Removed duplicate chart init and set default granularity to 1m.
- [frontend/src/services/api.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/services/api.js): Added `onPressureUpdate` and `pressure_update` event handler.
- [frontend/src/components/OrderBookPanel.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/OrderBookPanel.jsx): Converted pressure fetching to WebSocket.
- [frontend/src/components/SpecializedServicePanels.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/SpecializedServicePanels.jsx): Converted pressure panel to WebSocket.
- [services/serve/routers/retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/retrieval.py): Added `method` parameter to `/forecast`.
- [services/serve/routers/market.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/market.py): Added `/ws/pressure/{token}` WebSocket route.
- [services/serve/websocket.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/websocket.py): Added `pressure` channel to `ConnectionManager`.
- [src/cryptotrading/analysis/retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/analysis/retrieval.py): Supported `window_factor` and `historic_window_size`.
