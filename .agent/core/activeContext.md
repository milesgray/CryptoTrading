# Active Context: WebSocket Stream Unification & Fallback

## Quick Reference
- **Feature**: WebSocket-First streaming with automatic HTTP fallback
- **Branch**: `feature/websocket-unification`
- **Status**: Completed & Integrated ✅

## Executive Summary
Consolidated how the Vite React frontend retrieves live token price data. The App and subcomponents now leverage a unified, reference-counted WebSocket service (`webSocketService`). The connection is established dynamically when subscribers are active and falls back automatically to HTTP polling (`getLatestPrice`) during connection failures, reconnect phases, or server downtime.

## Key Files Created/Modified
- [api.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/services/api.js): Implemented reference counting subscriber tracker and HTTP fallback polling mechanism within `WebSocketService`.
- [App.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/App.jsx): Wired global Navbar info and selected token state to `webSocketService.onPriceUpdate` and eliminated duplicate HTTP polling.
- [CandlestickChart.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/CandlestickChart.jsx): Cleaned up manual socket disconnections.
- [OrderBookPanel.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/OrderBookPanel.jsx): Structured proper useEffect callbacks cleanup for the shared stream.

## Next Steps
- Verify frontend live updates with a running local instance.
