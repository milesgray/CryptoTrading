# Plan: SpecReTFForecaster Integration

## Goal
Implement the SpecReTF forecasting framework inside `services/retrieval/forecaster.py` by adding a new `SpecReTFForecaster` class.

## Proposed Components
1. **STFT Processing**: Partition 1D price sequences into overlapping windows, apply a Hann window, and compute FFT.
2. **Spectral Similarity**:
   - Amplitude similarity using normalized Jensen-Shannon Divergence.
   - Phase similarity using cosine of mean phase difference.
   - Composite frame score: sum of amplitude and phase similarity.
3. **Recency Weighting**: Apply exponential decay weighting to aggregate frame scores.
4. **Weighted Fusion**: Combine retrieved future continuations with direct query predictions.

## Execution Steps
1. Implement `SpecReTFForecaster` in `services/retrieval/forecaster.py`.
2. Implement unit tests in `tests/test_specretf.py`.
3. Update the fastapi startup / endpoint in `services/retrieval/main.py` if needed (or keep it compatible).
4. Run tests to verify correctness.
