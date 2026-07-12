# Active Context: Online Learning & Setup Archiver

## Quick Reference
- **Feature**: Online Learning & Setup Archiver
- **Branch**: `feature/retrieval-live-tracking`
- **Plan File**: `.agent/plans/online-learning-plan.md`
- **Status**: Execution Phase 🛠️

## Executive Summary
Implementing dynamic archiving of completed or partially completed forecast runs. When the forecast window completes (or is reset), the frontend sends the price returns and actual outcome to the serve proxy. The backend embeds the history, inserts the setup into PostgreSQL pgvector table, and rebuilds the retrieval index cache dynamically to increase search accuracy.

## Key Files to Modify
- [services/embed/server.py](file:///home/miles/Development/notebooks/CryptoTrading/services/embed/server.py): Add `/setup/add` endpoint.
- [services/retrieval/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): Add `/rebuild` endpoint to clear forecaster cache.
- [services/serve/routers/retrieval.py](file:///home/miles/Development/notebooks/CryptoTrading/services/serve/routers/retrieval.py): Add `/setup/add` proxy endpoint.
- [frontend/src/components/RetrievalVisualizer.jsx](file:///home/miles/Development/notebooks/CryptoTrading/frontend/src/components/RetrievalVisualizer.jsx): Hook up auto-save triggers and status alerts.

## Acceptance Criteria
- [ ] Embed service inserts a StoredTradeSetup into pgvector or numpy store.
- [ ] Retrieval service successfully flushes cache of the given symbol.
- [ ] Serve router exposes `/api/retrieval/setup/add` proxy.
- [ ] Frontend triggers setup archive when forecast window is completed or reset.
- [ ] Frontend displays dynamic database archival success alerts.



