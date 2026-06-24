# Active Context: Dedicated Forecasting Panel & Next Candle Predictor

## Quick Reference
- **Feature**: Dedicated Forecasting Panel, Next-Candle Color Classifier, and Configurable Ports
- **Branch**: `feature/forecast-visualization-and-predictions`
- **Status**: Completed & Verified ✅

## Executive Summary
Decouple forecasting projections from the main trading chart by building a specialized, full-width Pattern Matching & Retrieval Forecast panel. Integrate dynamic settings (number of segments, length, frequency, order book weight) and checkbox toggles for individual patterns. Render a high-impact Next Candle Color Predictor widget to classify the direction of the immediate next price move (GREEN/RED) with consensus confidence, and parameterize all service ports dynamically in the Docker Compose setup.

## Key Accomplishments
- **Clean Main Candlestick Chart**: Pruned `CandlestickChart.jsx` to remove all forecast overlays, states, and polling, returning it to a clean, highly optimized historical/live price candlestick visualizer.
- **Dedicated Retrieval Panel**: Rebuilt `RetrievalVisualizer.jsx` into a premium React dashboard component using ECharts to visualize recent query price history, individual retrieved patterns, and the calculated "Consensus Projection" line.
- **Interactive Settings & Toggles**: Added slider/select controls for segment count ($k$), segment length, frequency, and order book weight, along with color-coded checkbox toggles to show/hide individual retrieved sequences dynamically.
- **Next Candle Color Predictor**: Implemented a prominent indicator classifying the color of the very next candle (GREEN/RED) by calculating the consensus direction of the first forecasted step relative to the current close. Includes a horizontal progress bar showing consensus confidence (e.g. 80% of matches agree).
- **Consensus Metrics**: Added a summary statistics card displaying expected forecast return, bullish consensus ratio, uncertainty (forecast volatility), and average match strength.
- **Parameterize Compose Configurations**: Added configurable host port variables (for TimescaleDB, serve, retrieval, record, and frontend) to `.env` and `.env.example`, and updated `docker-compose.yml` to pull these configurations dynamically.
- **Fixed pgvector Type Registration**: Solved the `unknown type: pg_catalog.vector` warning by changing the schema in the connection's `set_type_codec` call from `'pg_catalog'` to `'public'` in `postgres.py`.

## Next Objectives
- Implement historical TimescaleDB compression and retention policies.
- Run database performance and latency load tests under simulated high-frequency updates.
- Integrate pgvector HNSW index queries into the forecasting logic.
