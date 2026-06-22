# Product Context: CryptoTrading

## User Personas

### Primary Users
1. **Quantitative Traders**
   - Execute manual or semi-automated trades based on the dashboard signals.
   - Monitor real-time price feeds, order books, and price pressure.
   - Use matched historical patterns to estimate trade outcome probabilities (long vs. short).
   - Use JEPA regime indicators to adapt active position leverage.

2. **Algorithm Developers / Quantitative Engineers**
   - Build automated trading bots integrated with the price servers.
   - Implement trading strategies using custom API hooks (REST or WebSockets).
   - Validate performance of calculated metrics (price pressure, order book imbalances).

3. **Machine Learning Researchers**
   - Design and train time-series encoders, forecasting models (TimesNet, Autoformer), and JEPA models.
   - Evaluate model embeddings, clustering quality (mean cluster purity), and predictor symmetry measures.
   - Backtest representations in perpetual futures trading simulators.

## Core User Needs

### Real-Time Market Data & Analysis
- **Accurate Index Pricing**: Low-latency, manipulation-resistant index prices built from raw spot and futures exchange order books.
- **WebSocket Streaming**: Real-time streaming of aggregated prices and transformed order book structures (spread, midpoint, bids, asks).
- **Price Pressure Metrics**: Quantitative metrics summarizing buying/selling pressure derived from order book depth.

### Pattern Matching & Regime Classification
- **Similar Setup Finder**: Immediate lookup of similar historical market patterns in pgvector to identify high-probability trade setups.
- **Market Regime Detection**: Dynamic classification of current price behavior (e.g. low-vol trending, high-vol range bound) to control risk.
- **Social Sentiment Correlation**: Live sentiment tracking of Twitter activity for relevant cryptocurrency tickers to feed predictive models.

### Model Training & Backtesting
- **Idiomatic ML Workflows**: Standard interfaces to train forecasting and movement models.
- **Simulation Environments**: Backtesting framework containing leverage controllers and perp futures dynamics (e.g. funding rates, bust/stop-loss triggers).

## User Journey Mapping

### Quantitative Research Workflow
1. **Data Gathering**: Access historical prices, order book snapshots, and Twitter tweets stored in MongoDB.
2. **Model Training**: Run the PyTorch training pipelines for SupCon setup encoders, Koopman-JEPA regime models, or TimesNet forecast models.
3. **Evaluation**: Examine model metrics (e.g. JEPA identity deviations, eigenvalues, cluster purity, validation loss).
4. **Deploy**: Push model weights to the live API services (`services/embed`, `services/jepa`).

### Real-Time Trading & Visualization Workflow
1. **Dashboard Start**: Launch Vite React dashboard connected to the FastAPI serving app.
2. **Stream Price**: Receive price updates, candlestick bars, and order book states via WebSockets.
3. **Pattern Ingestion**: View matched historical patterns and estimated profitability projections dynamically overlaid on TradingView charts.
4. **Exposure Adjustments**: Adjust leverage and sizes using JEPA-augmented indicators (trending vs high volatility).

## Business & Technical Requirements

### Functional Requirements
- High-frequency order book collection and composite indexing (500ms loops).
- WebSocket servers supporting dual subscriptions: `ws/price/{token}` and `ws/order_book/{token}`.
- REST endpoints for historical price queries, candlestick generation, and transformed order book statistics.
- pgvector HNSW database indexing and query capabilities.
- Sentiment analysis pipeline using Twitter developer streams.
- Model training arguments support for task types (forecast vs movement) and model choices (TimesNet, Autoformer, Transformer).

### Non-Functional Requirements
- Sub-500ms processing loop for price index aggregation.
- Fast embedding retrieval (<100ms) for similarity queries.
- Clean database isolation in test suites.
- Containerized deployments (Dockerfiles and docker-compose files) for easy local setup and cloud deploys.
