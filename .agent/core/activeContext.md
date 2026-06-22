# Active Context: CryptoTrading Memory Initialization

## Quick Reference
- **Active Task**: Aligning the Windsurf/Claude Code memory bank (`.agent/core/*.md`) with the CryptoTrading project realities.
- **Current Branch**: `main`
- **Key Modules**: `services/serve`, `services/price`, `services/jepa`, `services/embed`, `services/sentiment`, `frontend`.

## Executive Summary
We are setting up the documentation structure and memory bank for the CryptoTrading system, transitioning it from the Golden Age Hub template. The project is a microservice-based quantitative crypto framework integrating live composite indexing, self-supervised representation learning, pattern matching, and sentiment indexing.

## Key Files
- `services/serve/app.py`: FastAPI server containing WebSocket endpoints for price/order book streams.
- `services/price/service.py`: Logging and composite price calculator service.
- `services/jepa/model.py`: PyTorch Joint Embedding Predictive Architecture for market regimes.
- `services/embed/pipeline.py`: Contrastive CNN training and pgvector storage pipeline.
- `services/sentiment/analyzer.py`: Twitter sentiment scraper.
- `frontend/src/App.jsx`: Dashboards mapping token selection to candlestick charts and order book components.

## Next Steps
- Verify integration between the FastAPI serve application and the Postgres pgvector databases.
- Test connection pools under load with multiple active symbols.
- Assess the model inference latency for live setups matching.
