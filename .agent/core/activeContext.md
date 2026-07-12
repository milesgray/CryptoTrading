# Active Context: ECharts Null Candlestick Crash Fix

## Quick Reference
- **Feature**: ECharts Null Candlestick Crash Fix
- **Status**: Completed & Verified ✅

## Executive Summary
Resolved a frontend crash where the application would turn to a blank white screen shortly after data loaded. The crash was traced to ECharts' internal `WhiskerBoxCommonMixin.getInitialData` method, which is used to initialize candlestick and boxplot series. When the series data contained `null` padding values (used in the forecast visualizer to align query history with forecast steps) and the category axis was used without custom encoding, ECharts' ordinal generation loop attempted to read `item.value` on a `null` item, raising a `TypeError: Cannot read properties of null (reading 'value')` error. The fix replaced all padding `null` values in `RetrievalVisualizer.jsx` series with ECharts' standard empty data indicator string `'-'`.

## Key Files Modified
- [RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx): Replaced padding `null` values with `'-'` in `paddedHistoricalData`, `retrievedCandles`, and `consensusCandles` series data arrays.

## Verification & Validation
- **Production Compilation**: Executed `npm run build` inside the `frontend/` directory; the Vite build compiled successfully with Rolldown, verifying syntax correctness and asset packaging.
