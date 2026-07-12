# Feature Plan: Online Learning & Setup Archiver

## Objective
Update the forecasting pipeline to automatically save completed (realized) forecast runs to the database and clear the in-memory retrieval cache. This enables the pattern similarity matcher to match against the most recent actual market data dynamically.

## Proposed Changes

### 1. Embed Service (`services/embed/server.py`)
Add endpoint `POST /setup/add` with Pydantic request models:
- Normalizes and embeds the price window returns.
- Inserts `StoredTradeSetup` with 128D embedding into the PostgreSQL vector store.

### 2. Retrieval Service (`services/retrieval/main.py`)
Add endpoint `POST /rebuild`:
- Clear forecaster cache for the token.
- Forces rebuild of Annoy index on next query.

### 3. Serving Router (`services/serve/routers/retrieval.py`)
Add endpoint `POST /setup/add`:
- Proxies request to `services/embed` and `services/retrieval`.

### 4. Frontend Component (`RetrievalVisualizer.jsx`)
- Auto-save completed or partial runs (threshold: >=5 steps) on unmount, token change, or full completion.
- Show database status indicator in metrics panel.

## Verification
- Verify compilation and request flows end-to-end.
