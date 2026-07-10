# Active Context: Frontend Candlestick Query Chunking & Optimization

## Quick Reference
- **Feature**: Frontend Candlestick Query Chunking
- **Status**: Completed & Verified ✅

## Executive Summary
Optimized the retrieval of historical candlestick data on the frontend to avoid large queries that can lead to database timeouts and payload transmission issues. Developed a transparent range division algorithm directly in `api.js` that splits query durations into smaller granularity-based sub-ranges, queries them in small concurrent batches, gracefully handles empty chunk responses, and deduplicates and sorts the final consolidated data set.

## Architecture Overview
The frontend `getCandlestickData` function acts as a wrapper that:
1. Calculates chunk intervals based on a target size of 1000 data points per query (`chunkDurationMs = 1000 * granularity * 1000`).
2. Iterates over time ranges and requests chunks.
3. Limits concurrent request volume using a simple batch execution array (`CONCURRENCY_LIMIT = 3`).
4. Catches 404 response errors dynamically on missing ranges to return empty results rather than crashing the request.
5. Deduplicates by timestamp and sorts chronologically before returning the result array.

## Tech Stack for This Feature
- **React + Vite**: UI Rendering
- **Axios**: HTTP query execution & error handling
- **date-fns**: Time parsing and formatting

## Key Files Created/Modified
- [api.js](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/services/api.js): Implemented transparent chunking, batch concurrency, and deduplication logic.

## Verification & Validation
- **Automated Verification**: Ran a Node.js simulator test ([test_chunking.js](file:///home/miles/.gemini/antigravity-ide/brain/891add4f-aac1-4522-9c11-e282e89f7a1f/scratch/test_chunking.js)) verifying single-chunk, multi-chunk, parallel batching, sorting, and deduplication correctness.
- **Production Asset Build**: Executed `npm run build` inside `frontend/` successfully without compilation warnings/errors.

