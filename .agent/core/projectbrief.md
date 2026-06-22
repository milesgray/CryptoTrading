# Project Brief: CryptoTrading

## Context

CryptoTrading is an advanced, multi-service quantitative trading framework designed to process real-time cryptocurrency data, predict price movements, identify trading setup patterns, and classify market regimes using machine learning models. The framework specifically implements Rollbit's Futures Trading price formulation, contrastive learning encoders, and Joint Embedding Predictive Architectures (JEPA).

## Problem Statement

Traditional cryptocurrency trading frameworks often rely on simple indicators or single-exchange price feeds, which are susceptible to manipulation, latency, and noise. Traders need a platform that:
1. Calculates manipulation-resistant, real-time index prices across multiple spot and derivative exchanges.
2. Identifies historical trade setups in real-time.
3. Automatically classifies market regimes to adapt position sizing and leverage dynamically.
4. Integrates social media sentiment analysis (Twitter) into predictive models.

## Solution Overview

CryptoTrading addresses these needs with a modular, microservice-based architecture:
- **Rollbit Price Index Formulation**: Subscribes to spot and derivative exchanges via streaming APIs, filters stale/crossed feeds, aggregates limit orders into a composite order book (capping orders at $1M), and calculates a composite index price every 500ms using exponential weights and Viterbi path solvers.
- **Trade Setup Pattern Matcher**: Uses a contrastive CNN encoder (SupCon Loss) to map rolling price return windows into 128-dimensional embeddings stored in a PostgreSQL database using `pgvector` for real-time similarity search.
- **Koopman-JEPA Time Series Encoder**: Applies Joint-Embedding Predictive Architectures (JEPA) with a Koopman operator interpretation to discover dynamical regime indicators and adjust trading leverage dynamically.
- **Sentiment Analyzer Service**: Ingests Twitter data to compute sentiment scores for BTC, ETH, and other assets, saving results in MongoDB.
- **Deep Learning Forecasting Models**: Integrates timeseries forecasting architectures (TimesNet, Autoformer, Transformer) in PyTorch to predict price movement.
- **Vite React Frontend Dashboard**: Real-time visualization using TradingView Lightweight Charts, demonstrating live price streams, matched setups, and order book pressure.

## Goals

1. **Price Recording & Serving**: Deploy service components to connect to exchanges, compute the Rollbit index price, and serve data via WebSockets and FastAPI REST endpoints.
2. **Setup Matching Pipeline**: Train contrastive encoders and build the PostgreSQL/pgvector indexing system to match current setups against historical matches.
3. **Koopman-JEPA Integration**: Implement the self-supervised JEPA pipeline for regime classification and dynamic leverage controls.
4. **Sentiment Analyzer Deployment**: Deploy the containerized Twitter stream ingestion and sentiment scoring service.
5. **Real-time Frontend**: Run the React dashboard visualizing live price history, order books, and matched historical patterns.

## Success Metrics

- 500ms update intervals for composite index prices.
- Low-latency matching (<100ms) for historical setup patterns in pgvector.
- Clear regime separation (e.g., >60% mean cluster purity in JEPA vs ~38% in simple autoencoders).
- Accurate Twitter sentiment scoring saved to database in near real-time.
- Responsive React UI visualization using lightweight charting.

## Status

**Current**: Core microservices implemented (serve, price recording, sentiment analyzer, train, embed, jepa, pressure). Dockerfiles and docker-compose configurations are ready for serving, recording, and the React frontend.
**Next**: Finalize Postgres/pgvector setup for live embedding queries and verify model inference speed under load.

## Decision

The project uses a microservices architecture structured around FastAPI REST/WebSocket controllers, MongoDB for document/price-data storage (being migrated/augmented), and PostgreSQL + pgvector for setup embeddings. Pydantic is used for schema enforcement, and PyTorch provides the engine for neural networks (SupCon and JEPA).

## Alternatives Considered

- Monolithic Python application (rejected - performance bottlenecks and separation of concerns issues across model training, streaming APIs, and frontend).
- Traditional time-series models only (rejected - deep learning models like TimesNet/Autoformer and JEPA show superior representation learning of non-stationary crypto environments).

## Consequences

- **Benefits**: Modular scalability, independent training and prediction components, high reliability in data streams.
- **Trade-offs**: Multi-database management (MongoDB + PostgreSQL), higher orchestration overhead (Docker Compose / Dockerfiles).

## Implementation Status

- ✅ FastAPI / WebSocket Price Server (`services/serve`)
- ✅ Price Recording Service (`services/price`)
- ✅ Twitter Sentiment Analyzer Service (`services/sentiment`)
- ✅ Joint Embedding Predictive Architecture (JEPA) (`services/jepa`)
- ✅ Trade Setup Pattern Matcher with CNN and pgvector (`services/embed`)
- ✅ TimesNet, Autoformer, Transformer training script (`services/train`)
- ✅ Order Book Pressure/Features extractor (`services/pressure`)
- ✅ Vite React Frontend Dashboard (`frontend`)
- 🔄 Migration of legacy MongoDB price tracking to TimescaleDB/PostgreSQL (in progress)
