# Active Context: Chronos Retrieval Augmented Forecasting (RAF) Framework

## Quick Reference
- **Feature**: Chronos Retrieval Augmented Forecasting (RAF) Framework
- **Plan File**: [implementation_plan.md](file:///home/miles/.gemini/antigravity-ide/brain/3dc38ad7-a3d9-403d-9992-a7c847eb63a6/implementation_plan.md)
- **Status**: Completed ✅

## Executive Summary
Implementing the paper's Retrieval Augmented Forecasting (RAF) framework in the retrieval service, utilizing the existing `ChronosPipeline` to generate similarity-augmented forecasts. We implement the complete RAF workflow: retrieving top-k segments, separately normalising the query and retrieved patterns, aligning boundary joins to enforce continuity, and running zero-shot forecasting on the augmented series before denormalizing back to the original price space.

## Tech Stack for This Feature
- **Python / FastAPI**: Core forecasting microservice logic.
- **PyTorch / Transformers**: Chronos tokenization and generation inference.
- **NumPy**: Data structures, normalization, and continuity offsets.

## Key Files Created/Modified
- [services/retrieval/forecaster.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/forecaster.py): Implemented the `ChronosRAFForecaster` class.
- [services/retrieval/main.py](file:///home/miles/Development/notebooks/CryptoTrading/services/retrieval/main.py): Loaded the pre-trained `ChronosPipeline` on startup, cached forecasters dynamically by symbol, and updated `/forecast` to support `method="raf"`.
- [tests/test_raf_forecaster.py](file:///home/miles/Development/notebooks/CryptoTrading/tests/test_raf_forecaster.py): Wrote unit tests confirming offset alignment, normalization, and predictions.

## Critical Implementation Details
1. **Separate Normalization**: Query context and retrieved segments are normalized separately to mitigate distribution shifts.
2. **Boundary Continuity**: Additive offset is computed as `z_orig[0] - z_ret[-1]` to shift the retrieved sequence and eliminate visual/mathematical jumps at the boundary.
3. **輕量級 Chronos**: Defaults to `amazon/chronos-t5-mini` which has a ~35MB footprint and fast CPU generation times, perfect for robust zero-shot forecasting.

## Next Steps
- Verify the new RAF predictions render in the live frontend candlestick consensus chart series.
