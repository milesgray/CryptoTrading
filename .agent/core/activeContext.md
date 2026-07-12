# Active Context: Retrieval Forecast Sizing Optimization

## Quick Reference
- **Feature**: Retrieval Forecast Sizing Optimization
- **Status**: Completed & Verified ✅

## Executive Summary
Optimized the scaling of retrieved cycles and consensus projections in the forecasting panel. Previously, they were scaled using a hardcoded 1.5% volatility multiplier relative to the terminal query price (e.g. ~960 USD for a 64k BTC price), which led to an overly expanded vertical chart axis. This squished the preceding historical candles into tiny, unreadable lines. The fix calculates the actual standard deviation of the preceding (query) candle price series dynamically and scales the retrieved segments accordingly. Added safety margins (floor at 0.05% of price, ceiling at 2.0% of price) to ensure clean visualization in flat or highly volatile periods.

## Key Files Modified
- [RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx): Replaced hardcoded `0.015` multiplier with dynamic `getScaleMultiplier` based on preceding query prices standard deviation.

## Verification & Validation
- **Production Compilation**: Executed `npm run build` inside the `frontend/` directory; the Vite build compiled successfully.
