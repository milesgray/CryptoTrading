# Implementation Plan: Real-Time Token Pressure Calculation

## Executive Summary
Replace static token pressure values in `services/pressure/main.py`'s `get_token_pressure` handler with real-time orderbook fetching, feature extraction via `OrderBookFeaturizer`, and inference using `PressureModel`.

## Key Changes
1. **`services/pressure/main.py`**:
   - Initialize `OrderBookDataLoader` on startup.
   - Update `get_token_pressure(token: str)` to load recent orderbook snapshot for `token` (or generate/fallback if unavailable).
   - Featurize snapshot via `OrderBookFeaturizer`.
   - Run model prediction for pressure outputs.
2. **Tests**: Add unit/integration tests verifying real-time calculation.

## Acceptance Criteria
- `GET /{token}` calculates and returns dynamic features and model predictions instead of static values (`cvd: 2500`, `bap: 50.0`, etc.).
