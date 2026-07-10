# Task Log: Dynamic Retrieval Forecasting

## Task Information
- **Date**: 2026-07-10
- **Time Started**: 18:32
- **Time Completed**: 18:35
- **Files Modified**:
  - [main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py)
  - [retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/retrieval.py)
  - [RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx)

## Task Details
- **Goal**: Add dynamic granularity (frequency) and segment length (window size) options to the pattern-matching retrieval forecasting dashboard and hook them up to the React frontend.
- **Implementation**:
  - Built a dynamic index manager and caching scheme in `main.py` using `asyncio.Lock` to thread-safely compile custom Annoy vector indices on first query.
  - Dynamically scaled STFT parameters, FFT bins, and retrieval durations based on target granularity and segment length configurations.
  - Passed `granularity` and `window_size` parameters from the React frontend to the serve gateway router, increasing client timeout to 30.0s to allow initial index compilation.
  - Dynamically recalculated baseline price bounds and formatted ECharts labels/alignments to match active steps and frequencies.
- **Challenges**:
  - Handling situations where historical DB candlestick counts were smaller than `window_size + horizon` (solved by dynamically scaling the horizon down or showing a clear warning).
- **Decisions**:
  - Implemented dynamic cache keys containing the currency token, granularity, and segment length to support asset-specific pattern search.

## Performance Evaluation
- **Score**: 22/23
- **Strengths**:
  - Optimized resource usage by compiling indices only on-demand instead of pre-computing all 16 permutations at startup.
  - Ensured complete thread-safety during concurrent index builds.
  - Retained strict backward compatibility with existing tests and defaults.
- **Areas for Improvement**:
  - None, implementation succeeded gracefully on all fronts.

## Next Steps
- Monitor frontend behavior in production configurations under higher concurrent user loads.
