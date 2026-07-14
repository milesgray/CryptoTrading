# Feature Plan: Lazy Candlestick Chart Loading & Overlay Bug Fix

## 1. Feature Overview
- **Description**: Make the main candlestick chart default to 5-minute granularity, load the visible chart range (200 candles) plus an additional 200 candles preemptively on startup, and dynamically/lazily load older historical data in the background as the user pans or zooms out. Also, fix the alignment/overlap bug in the pattern matching retrieval forecast chart.
- **Goals**:
  - Main chart defaults to 5-minute granularity.
  - Efficiently fetch and display data (reducing database query size by starting with 400 candles instead of a full week).
  - Smooth, background lazy-loading of older historical data during panning/zooming.
  - Zero overlaps between historical candles and forecast segments in the retrieval forecast chart.
- **Scope Boundaries**:
  - **IN**: Main candlestick chart component, pattern matching retrieval visualizer frontend component.
  - **OUT**: Backend API router and database storage changes (existing endpoints support the requested granularity and pagination query params).

## 2. Architecture & Design
- **Pattern(s) Used**: Client-side state synchronization, event-driven data pagination, asynchronous background requests.
- **Component Structure**:
  - `CandlestickChart.jsx` will manage data-table references, loaded range references, and background fetching state.
  - `RetrievalVisualizer.jsx` will base series coordinates strictly on the real size of active data arrays to avoid state misalignment during transitions.

## 3. Technical Implementation
- **Tech Stack**: React, AnyChart (stock chart), ECharts (for forecast).
- **Key Algorithms/Patterns**:
  - AnyChart `xScale` listener (`propertyChange` matching `minimum`) for panning detection.
  - Chronological merging/deduplication of paginated candlestick chunks using `Map` keys.

## 4. Acceptance Criteria
- [ ] Main candlestick chart defaults to `5 Minutes` granularity on load.
- [ ] Initial data payload is limited to 400 candles.
- [ ] Default view zoom is centered on the last 200 candles.
- [ ] Panning left beyond the 50-candle threshold triggers a silent background fetch of the next 400 candles.
- [ ] No full-screen loading spinner is displayed during history panning.
- [ ] Changing retrieval settings (e.g., segment length) never causes overlapping series or mismatched time ticks.

## 5. Task Breakdown
- **Task 1**: Modify `CandlestickChart.jsx` default states and initial fetching bounds.
- **Task 2**: Implement AnyChart `xScale` listener and the background chunk loader `fetchMoreHistory`.
- **Task 3**: Update `RetrievalVisualizer.jsx` series logic to use `queryCandles.length` for positioning.
- **Task 4**: Verify all features visually on the dev server.

## 6. Risk Assessment
- **Risk**: Repeated background requests if a network request fails or if user remains past threshold.
  - *Mitigation*: Protect fetch with `isFetchingHistoryRef.current` lock and move the minimum loaded range timestamp back even if empty data or errors occur to prevent infinite loops.
- **Risk**: Chart shifting or resetting focus when data is prepended.
  - *Mitigation*: Prepending data before the visible range start key should not affect the current visible range keys in AnyChart.
