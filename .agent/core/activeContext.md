# Active Context: Lazy Candlestick Loading & Overlay Alignment Fix

## Quick Reference
- **Feature**: Lazy Candlestick Loading & Overlay Alignment Fix
- **Branch**: `feature/lazy-candlestick-loading`
- **Plan File**: `.agent/plans/lazy-candlestick-loading-plan.md`
- **Status**: Completed ✅

## Executive Summary
Optimizing the main candlestick chart to load the last 200 visible candles (plus 200 preemptively) at a default of 5-minute granularity on startup. Dynamic background history fetching is triggered when the user pans near the loaded historical limit using AnyChart's xScale listener. We also resolve the overlapping/misaligned series bug in the Retrieval Forecast visualizer by using the actual query dataset length instead of transient setting state. Finally, we resolved an "Embedding dimension != index dimension" ValueError in the retrieval service by dynamically querying the active embedding dimension from the embed service health endpoint and calculating local handcrafted dimensions dynamically based on window_size.

## Tech Stack for This Feature
- **React**: Component UI state, effect synchronization, and event binding.
- **AnyChart (Stock)**: Main price charting, dynamic data tables, and scale property change listeners.
- **ECharts**: Forecast matching overlay rendering.
- **Python / FastAPI**: Core embedding and pattern matching backend.

## Key Files to Create/Modify
- [frontend/src/components/CandlestickChart.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/CandlestickChart.jsx): Change default granularity, initial range calculation, call selectRange to focus the view, and listen to xScale change events to trigger background fetches.
- [frontend/src/components/RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx): Use `queryCandles.length` instead of state-based `segmentLength` in series rendering arrays to align series correctly and eliminate overlaps.
- [services/retrieval/encoder.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/encoder.py): Determine the embed service dimension dynamically from `/health` and calculate combined dimension sizes based on active parameters.
- [services/retrieval/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): Dynamically calculate and pass combined dimensions when initializing vector indexes.

## Critical Implementation Details
1. **Background Pagination**: Use xScale `propertyChange` checking `minimum` to dynamically trigger fetches without resetting full-screen loader states.
2. **Data Deduplication**: Keep loaded data elements unique by indexing with millisecond timestamps and sorting chronologically before updating the AnyChart data table.
3. **Decoupled Render Scales**: Align series shapes on the forecast chart to the actual baseline length in the local scope rather than the global configuration state.
4. **Dynamic Dimension Matching**: Concatenating neural embeddings with handcrafted local features requires dynamically computing local spectral/orderbook feature shapes using `n_fft` parameters to set the correct Annoy index dimensions.

## Acceptance Criteria
- [x] Main candlestick chart defaults to 5-min granularity.
- [x] Loads 400 candles on startup but zooms to the last 200 candles.
- [x] Panning left loads previous chunks silently in the background.
- [x] Retrieval forecast chart has zero overlap between historical series and projections.
- [x] Retrieval service dynamically computes index dimensions to prevent ValueError crashes on window size changes.
- [x] Fix synchronous blocking calls in FastAPI services (main.py and encoder.py).
- [x] Fix frontend request flood by implementing a 10s cooldown upon candlestick data fetch failures.

## Next Steps
- Verify the fixes in a live development setup.
- Merge the Pull Request `#69` into main.
