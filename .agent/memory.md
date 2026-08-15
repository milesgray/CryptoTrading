# Project Memory

## Last Completed Task
Fixed duplicate AnyChart chart rendering, changed default chart granularity to 1m, added `method` parameter support (`raf`, `specretf`, `retrieval`) through serve service to retrieval forecasting endpoint, converted order book pressure frontend data delivery to real-time WebSockets (`/ws/pressure/{token}`), and updated `RetrievalEncoder` with backward-compatible `window_factor` support.

## Architecture Notes
- Serve service proxies forecasting requests to retrieval service including `method` (`raf`, `specretf`, `retrieval`).
- WebSocket manager in `services/serve/websocket.py` supports `price`, `order_book`, and `pressure` channels.
- Frontend `WebSocketService` in `frontend/src/services/api.js` subscribes to real-time `pressure_update` events with automatic HTTP polling fallback.
- `RetrievalEncoder` supports both `historic_window_size` and `window_factor`.

## Environment / Config
- None.

## Dependencies Added
- None.

## Known Blockers / Next Steps
- None.
