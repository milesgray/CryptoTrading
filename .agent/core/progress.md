# Progress: CryptoTrading

## Project Timeline

### Phase 1: Foundation & Data Streaming (Completed ✅)
- [x] uv environment configuration and dependencies (`ccxt`, `motor`, `pymongo`, `fastapi`, `torch`).
- [x] Rollbit Price System composite index price formulation (500ms intervals, multi-exchange parsing, outlier removal, order size caps, exponential weightings).
- [x] Ingestion and recording service (`services/price`) running with graceful shutdown signals.
- [x] FastAPI REST/WebSocket serving application (`services/serve`) supporting:
  - WebSocket connections (`/ws/price/{token}`, `/ws/order_book/{token}`).
  - REST query handlers (`/candlestick/{token}`, `/historic/price/{token}`, `/health`).
- [x] Docker setup (`Dockerfile.serve`, `Dockerfile.record`, `/frontend/Dockerfile`, `docker-compose.yml`).

### Phase 2: Representation Learning & AI Models (Completed ✅)
- [x] **Trade Setup Pattern Matcher** (`services/embed`): CNN-based SupCon encoder mapping log return price windows to 128-dimensional vectors.
- [x] **Koopman-JEPA Regime Classifier** (`services/jepa`): Self-supervised Joint-Embedding Predictive Architecture utilizing Koopman invariants for unsupervised market regime identification.
- [x] **Twitter Sentiment Analyzer** (`services/sentiment`): Containerized stream ingestion service scoring tickers using VADER.
- [x] **TimesNet/Autoformer Trainer** (`services/train`): Configurable forecasting and movement classification trainer in PyTorch.

### Phase 3: Frontend Dashboard & Visualization (Completed ✅)
- [x] Vite React frontend layout (`frontend/src/App.jsx`).
- [x] Lightweight Charts implementation for Candlestick visualization.
- [x] Real-time order book panel listening to WebSocket streams.

### Phase 4: Database Integration & Production Simulation (Completed ✅)
- [x] Migrate primary database backend from MongoDB to PostgreSQL with TimescaleDB (Postgres as new default, Mongo as option).
- [x] Connect the FastAPI server directly to the PostgreSQL + pgvector setups library for live pattern matching queries.
- [x] Implement TimescaleDB migrations for high-frequency pricing historical databases.
- [x] Build dedicated ECharts retrieval-augmented forecasting panel and next-candle direction predictor.
- [x] Run load tests evaluating API WebSocket server latency across multiple token configurations.

### Phase 5: Deep Learning Training Pipeline & RAFT Integration (Completed ✅)
- [x] Modify timeseries dataset and dataloaders to return absolute sample indices.
- [x] Resolve Python 3 relative/absolute import pathways and case-sensitivity issues in the deep learning submodules.
- [x] Extend `ForecastExp` and `MovementExp` training runners to support RAFT pre-computation phases and index-based forward signatures.
- [x] Fix training loop evaluation bugs and implement dynamic evaluation modes in `forecast.py` and `movement.py`.
- [x] Write and run training pipeline integration tests verifying deep learning training pipelines end-to-end.

### Phase 6: Order Book Analytics & Multi-Exchange Metrics (Completed ✅)
- [x] Fix bid/ask filter logic bug in `validate_feeds` in `book.py`.
- [x] Add detailed mathematical docstrings to the Rollbit index price calculation in `formula.py`.
- [x] Implement multi-exchange order book meta-statistics (price dispersion, HHI concentration, global arbitrage spreads).
- [x] Write and verify comprehensive unit tests in `tests/test_order_book_analytics.py`.

### Phase 7: FastAPI Codebase Refactoring & Modularity (Completed ✅)
- [x] Modularize `services/serve/app.py` by extracting endpoints to specific APIRouters under `services/serve/routers/`.
- [x] Extract market feed REST/WebSocket streams to `market.py`.
- [x] Extract retrieval forecasting proxy to `retrieval.py`.
- [x] Extract background service control and status WebSocket endpoints to `services.py`.
- [x] Resolve circular imports by shifting dependencies to request/websocket contexts.
- [x] Instantiate a unified global `websocket_manager` singleton in `websocket.py`.

### Phase 8: Historical Price Ingestion Optimization & Caching (Completed ✅)
- [x] Optimize CCXT historical price pulling by reusing active connections.
- [x] Implement a dual-layer caching system (PostgreSQL TimescaleDB + local JSON file fallback).
- [x] Support time-based progress bar and download cancellation/resumption.
- [x] Implement smart active symbol matching to automatically map tokens to available trading pairs.
- [x] Build and verify unit tests in `tests/test_exchange.py`.

### Phase 9: SpecReTF Forecasting Framework Integration (Completed ✅)
- [x] Implement `SpecReTFForecaster` incorporating STFT, Jensen-Shannon Divergence, and amplitude-weighted phase coherence.
- [x] Build dual-pathway forecasting pipeline (retrieval consensus + direct query path) with heuristic and model-weighted fusion.
- [x] Verify mathematical correctness and trend identification using unit tests in `tests/test_specretf.py`.

### Phase 10: WebSocket Stream Unification (Completed ✅)
- [x] Enhance `WebSocketService` with subscription tracking and automatic connection lifecycle sync.
- [x] Implement automatic HTTP polling fallback for connection dropouts and handshake delays.
- [x] Clean up component-specific manual disconnections to allow shared pricing connection states.

### Phase 11: Leveraged DP Oracle Optimization & Testing (Completed ✅)
- [x] Correctness validation fixes for NaN and negative price checks in Numba kernel.
- [x] Mathematical and inner-loop optimizations inside dynamic programming engine.
- [x] Caching object-address retention fix to prevent memory reuse misses.
- [x] Comprehensive unit tests verifying oracle behaviors, compounding, NMS, caching, and NaNs.

### Phase 12: HTTP Enabled Training Service & Frontend (Completed ✅)
- [x] Convert train service from CLI-driven to FastAPI-based HTTP server.
- [x] Support asynchronous training background tasks.
- [x] Add status monitoring, model listing, and weight download endpoints.
- [x] Integrate multi-axis inference serving endpoint for all `cryptotrading.predict` models.
- [x] Fix UnboundLocalError bug in forecast experiment trainer loss logic.
- [x] Configure Vite proxying and environment variable networks for frontend container.
- [x] Implement Axios service wrappers and update React ModelTrainingConsole.
- [x] Build live training job queue status tracker, dynamic loss curves, and checkpoint inference tester.
- [x] Verify successful production asset build using Vite compiler.

### Phase 13: Embed Service Polish & pgvector Centralization (Completed ✅)
- [x] Modify `pgvector_store.py` to support native DSN parameters.
- [x] Update `server.py` startup, REST, and WebSocket logic to pull from PostgreSQL/pgvector.
- [x] Correct port binding from 8000 to 8301 to match docker-compose configuration.
- [x] Migrate pgvector database layer to the main `cryptotrading` package.
- [x] Centralize `trade_setups` schema and index generation in `postgres.py`.
- [x] Write and run unit tests for all touched components.

### Phase 14: Service Communication & Embed Auto-Population (Completed ✅)
- [x] Configure `EMBED_SERVICE_URL` in docker-compose configs to resolve container communication.
- [x] Implement async background task to automatically extract trade setups, generate embeddings, and populate pgvector database at startup.
- [x] Verify complete container lifespan, communication flows, and test execution.

### Phase 15: Database Query Optimization & Restoring Candlestick Charts (Completed ✅)
- [x] Recover host disk space (pruning Docker resources to free ~13.2 GB) allowing TimescaleDB database container to boot successfully.
- [x] Resolve SQL query timeouts on the 9.5M-row `price_data` table by replacing slow JSONB filters and pattern-LIKE matches with SkipScan-based `resolve_matching_symbols` and indexed `symbol = ANY($1)` scans.
- [x] Fix candlestick query timeouts on the remote server by adding a composite index on `(symbol, exchange, time DESC)` and disabling statement timeouts during schema initialization to allow large-table DDL updates on startup.
- [x] Verify query performance and correctness via tests and endpoints, restoring candlestick rendering in under 200 ms.

### Phase 16: Frontend Candlestick Query Chunking (Completed ✅)
- [x] Chunk up large historical candlestick fetches transparently inside `getCandlestickData` using dynamic range division based on query granularity.
- [x] Implement a concurrent batching queue (with limit of 3 concurrent requests) to execute chunk queries in parallel safely.
- [x] Deduplicate overlapping data points by timestamp and sort chronologically.
- [x] Handle 404 response cases gracefully by treating missing data chunks as empty lists instead of failing the entire query.

### Phase 17: Dynamic Retrieval Granularity Forecasting & Self-Healing (Completed ✅)
- [x] Implement thread-safe dynamic in-memory vector index cache in retrieval service.
- [x] Configure dynamic STFT parameters (frame size, hop size, FFT bins, forecast horizon) based on target window size.
- [x] Expose granularity and window size query parameters on the serve proxy and microservice.
- [x] Hook up React frontend frequency and segment length controls to trigger auto-updates on value change.
- [x] Enable dynamic date range calculation and chart labels scaling on the frontend.
- [x] Implement self-healing auto-training logic and dynamic PyTorch DataLoader batch scaling inside the embed service.
- [x] Configure persistent Docker volume mounts for encoder checkpoints.
- [x] Convert the retrieval forecasting visualizer to a multi-series candlestick chart.

## Sprint Progress
 
- [x] Add granularity to retrieval forecast and hook it up to the frontend.
- [x] Remove mock results and simulated data states from the frontend.
- [x] Fix dropdown and input styling issues (white-on-white text background) globally.
- [x] Align CandlestickChart and OrderBookPanel components with the global dark theme.
- [x] Rework Order Book snapshot panel into a cumulative depth chart visualizer.
- [x] Polish up the embed service, centralize pgvector database storage, and write comprehensive tests.
- [x] Implement transparent historical candlestick query chunking on the frontend.
- [x] Resolve the embed service missing model weights issue and ensure proper embedding comparisons.
- [x] Implement backend database query chunking for candlestick data retrieval to prevent statement timeouts.
- [x] Implement ECharts-based candlestick chart in the retrieval visualizer including historical baseline, lavender consensus projection, and transparent cyan retrieved matches.

### Phase 18: Batch Embedding Optimization & Boundary Aggregation Fix (Completed ✅)
- [x] Implement batched embedding endpoint (`POST /embed/batch`) on the embed service.
- [x] Integrate bulk segment indexing (`add_segments_batch`) chunked in sub-batches of 1000 in retrieval service.
- [x] Fix candlestick aggregation boundary logic in `src/cryptotrading/data/price.py` and `services/serve/data.py` to correctly map time buckets.
- [x] Rebuild and redeploy containers on the remote host, completing bootstrapping in under 2.5 minutes.

### Phase 19: Candlestick Volume Aggregation Fix (Completed ✅)
- [x] Correct the calculation of volume for historical candlestick data by summing the order book snapshot volumes of all ticks in a candle's time window, matching the behavior of the live WebSocket updates.
- [x] Apply changes to MongoDB and PostgreSQL adapters in `src/cryptotrading/data/price.py` and `services/serve/data.py`.
- [x] Verify correct non-zero sum of tick volumes on historic queries.

### Phase 20: ECharts Candlestick null Data Crash Fix (Completed ✅)
- [x] Fix the frontend blank white page crash by replacing padding `null` values with ECharts-compliant placeholder string `'-'` in the forecasting candlestick chart series.

### Phase 21: Retrieval Forecast Sizing Optimization (Completed ✅)
- [x] Calculate the preceding query price series standard deviation dynamically.
- [x] Scale the retrieved historical cycles and consensus projection using the dynamic standard deviation instead of a hardcoded 1.5% multiplier.
- [x] Enforce a minimum multiplier floor (0.05% of price) and maximum ceiling (2.0% of price) to handle extremely flat or volatile segments.
- [x] Resolve visual squishing on the y-axis, ensuring both preceding and retrieved candles are displayed at a similar, readable size.

### Phase 22: Retrieval Chart Overlap Resolution (Completed ✅)
- [x] Convert individual retrieved pattern segments from overlapping candlestick series to smooth line series.
- [x] Align and connect the line series to start seamlessly at the last price point of the historical candlestick series.
- [x] Synchronize the pattern line colors on the chart with the colors specified in the sidebar legend.
- [x] Keep the Consensus Projection as a candlestick series to preserve the bullish/bearish candle direction visual cues.

### Phase 23: Retrieval Chart Live Tracking (Completed ✅)
- [x] Hook the Pattern Matching & Retrieval Forecast chart into the WebSocket price update stream.
- [x] Slice incoming price ticks into forecast steps and render a growing actual price line in the forecast window.
- [x] Style the actual price line with a neon rose glow (#f43f5e) and dynamic endpoint marker.
- [x] Implement a Live Performance Tracking card displaying elapsed steps, price error, and direction matching status (Confirming vs Diverging).
- [x] Reset and clear live tracking data automatically when new forecast queries are run.

### Phase 24: Online Learning & Setup Archiver (Completed ✅)
- [x] Add `/setup/add` endpoint to embed service (`services/embed/server.py`) to dynamically insert new setups with 128D embeddings into pgvector.
- [x] Add `/rebuild` endpoint to retrieval service (`services/retrieval/main.py`) to invalidate and flush cached forecasters by symbol.
- [x] Implement `/setup/add` proxy endpoint in API serve router (`services/serve/routers/retrieval.py`) to process and route setups and trigger cache rebuilds.
- [x] Implement robust frontend lifecycle auto-archiving (`RetrievalVisualizer.jsx`) supporting full completion saves, interruption saves, and unmount cleanups via refs.
- [x] Add real-time database sync status badges (`Archiving...`, `Archived ✅`, `Failed ❌`) to the live performance tracking card.

### Phase 25: Chronos Embedding Integration (Completed ✅)
- [x] Integrate Amazon Chronos (`chronos-t5-base` in `bfloat16`) to the embed service pipeline and server.
- [x] Fix invalid `ChronosPipeline` constructor call to use `ChronosPipeline.from_pretrained(...)`.
- [x] Implement a unified `generate_embedding(self, x)` in `TradePipeline` and fallback in `AppState` to support both CNN encoder and optional Chronos embeddings (combined 896D).
- [x] Add cast to `.float()` before NumPy conversion to support `bfloat16` tensors on CPU and fix copy-construction warning.
- [x] Replace all direct `state.encoder` embedding calls in endpoints with unified `state.generate_embedding` to ensure correct Chronos integration.
- [x] Document `"use_chronos"`, `"chronos_model_id"`, and `"chronos_torch_dtype"` configuration parameters and their embedding dimension consequences in `README.md`.
- [x] Pass all pytest test cases in `tests/test_embed_service.py` and `tests/test_pgvector_store.py`.

### Phase 26: Lazy Candlestick Loading & Overlay Alignment Fix (Completed ✅)
- [x] Make the main candlestick chart default to 5-minute granularity.
- [x] Load 400 candles initially but default zoom to the last 200 candles.
- [x] Implement lazy background fetching of historical candlestick chunks when panning near the loaded boundary.
- [x] Fix the transition misalignment/overlapping bug in the retrieval forecast chart.
- [x] Fix the "Embedding dimension != index dimension" ValueError by dynamically calculating dimensions based on n_fft and embed service configuration.

### Phase 27: Chronos Retrieval Augmented Forecasting (RAF) Framework (Completed ✅)
- [x] Implement `ChronosRAFForecaster` in `forecaster.py` supporting separate instance normalization, join continuity offset alignment, sequence concatenation, and denormalized consensus outputs.
- [x] Integrate global Chronos pipeline loading on startup and dict-based caches for SpecReTF and RAF forecaster models.
- [x] Update forecast FastAPI endpoint to support configurable `method` parameter (defaulting to `"raf"`).
- [x] Create `test_raf_forecaster.py` and verify all test suites succeed.

### Phase 28: TimescaleDB Performance Optimizations & Downsampling (Completed ✅)
- [x] Cache SkipScan-based symbol resolution in `postgres.py` to eliminate 9-second fetch overhead on subsequent reads.
- [x] Optimize order book and price queries in `book.py` and `price.py` to use exact single-symbol lookup matching (`symbol = $1`) when possible to leverage Index Scan Backward and eliminate Sort steps.
- [x] Implement downsampling in `book.py` using TimescaleDB's `time_bucket` and `last()` aggregate functions.
- [x] Pass the expected interval in `data_loader.py` to allow database-side downsampling.
- [x] Benchmark optimizations verifying speedups up to 4x on DB execution and 2.5x on downsampled fetches.

### Phase 29: Dockerfile Optimization & PyTorch CPU Wheels (Completed ✅)
- [x] Split heavy deep learning dependencies into an optional `ml` extra group in `pyproject.toml` to prevent pulling PyTorch/CUDA in core services.
- [x] Configure explicit `pytorch-cpu` wheel index url pointing to `download.pytorch.org` to fetch optimized, lightweight CPU-only PyTorch.
- [x] Regenerate `uv.lock` local lockfile.
- [x] Modify all microservices Dockerfiles to only install necessary extra dependency groups (`--extra ml` where needed, raw sync for core services).
- [x] Rebuild and run all containers sequentially on the remote server to prevent disk/CPU thrashing, successfully reducing compressed image sizes by up to 85%.
- [x] Fix the symbol resolution database cache invalidation race condition inside `postgres.py` by implementing auto-refresh on missing symbol lookups.

### Phase 30: Retrieval Forecast Window Alignment & Timeout Tolerance (Completed ✅)
- [x] Fix HTTP 400 Bad Request error in retrieval forecast due to initial startup gaps by increasing the price data query lookback window multiplier from 2 to 12.
- [x] Fix HTTP 502 Bad Gateway timeout error in serve proxy by increasing the HTTPX client timeout from 30 seconds to 90 seconds to tolerate Chronos forecasting CPU inference latency.

### Phase 31: FastAPI Pressure Service & Frontend Integration (Completed ✅)
- [x] Create FastAPI application wrapper (`main.py`) in `services/pressure/` with endpoints `/features` and `/predict`.
- [x] Integrate asynchronous model training via `POST /train` (background task) and progress polling via `GET /train/status`.
- [x] Fix relative imports in pressure submodules to support direct top-level absolute imports for container deployment and pytest.
- [x] Define `VITE_PRESSURE_URL` proxy and environment settings in Vite and docker-compose configurations.
- [x] Add dynamic model training console UI, parameter adjustments, and progress tracking inside the frontend's Order Book Pressure panel.
- [x] Run and pass all tests inside `services/pressure/test_data_loader.py`.

### Phase 32: TimescaleDB Candlestick & Order Book Optimizations (Completed ✅)
- [x] Create multi-timescale continuous aggregates (`price_candle_1s`, `price_candle_15s`, `price_candle_30s`, `price_candle_1m`, `price_candle_5m`, `price_candle_1d`) and refresh policies in `postgres.py`.
- [x] Optimize `get_candlestick_data` in `price.py` to route to continuous aggregates or fall back to SQL-side `time_bucket` grouping.
- [x] Implement database-side JSONB slicing using `jsonb_path_query_array` in `get_orderbook_data` in `book.py`.
- [x] Add lightweight `get_orderbook_summary` query in `book.py`.
- [x] Verify database adapter query executions successfully.
