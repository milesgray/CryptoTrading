# Active Context: Dockerfile Optimization & PyTorch CPU Wheels

## Quick Reference
- **Feature**: Dockerfile Optimization & PyTorch CPU Wheels
- **Plan File**: [implementation_plan.md](file:///home/miles/.gemini/antigravity-ide/brain/f7b7d684-c9cd-4b42-88eb-4b7003e6605d/implementation_plan.md)
- **Status**: Completed ✅

## Executive Summary
Optimized container builds and image sizes to prevent remote server CPU/disk bottlenecks and container crash-loops. By splitting heavy ML dependencies from core dependencies in `pyproject.toml` and configuring `tool.uv` sources to fetch PyTorch CPU-only wheels (`+cpu`), we reduced non-ML service images (like `record` and `serve`) by 85% (~1.4GB down to 330MB compressed) and ML service images (like `retrieval`, `predict`, `embed`, `train`) by 70% (~9.45GB down to 635MB compressed).

## Tech Stack for This Feature
- **Docker / BuildKit**: Container builds, caching, and multi-stage builds.
- **Astral uv**: Fast dependency management, lockfile synchronization, and optional dependency extra groups (`--extra ml`).
- **PyTorch (CPU-only)**: Optimized PyTorch CPU wheels (`+cpu`) for non-GPU VM host environments.

## Key Files Modified
- [src/pyproject.toml](file:///home/miles/Development/notebooks/CryptoTrading/src/pyproject.toml): Split dependencies into core and optional `ml` group; configured the `pytorch-cpu` package index.
- [src/uv.lock](file:///home/miles/Development/notebooks/CryptoTrading/src/uv.lock): Regenerated lockfile without CUDA dependencies.
- [Dockerfile.retrieval](file:///home/miles/Development/notebooks/CryptoTrading/Dockerfile.retrieval) & ML service Dockerfiles: Configured `uv sync` to compile using the `--extra ml` flag.
- Core service Dockerfiles (`Dockerfile.record`, `Dockerfile.serve`, `services/trade/Dockerfile`): Maintained lightweight sync without `ml` dependencies.
- [src/cryptotrading/data/postgres.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/data/postgres.py): Added cache invalidation auto-refresh on `resolve_matching_symbols` to resolve the startup bootstrap race condition.

## Critical Implementation Details
1. **CPU PyTorch Package Index**: Added `https://download.pytorch.org/whl/cpu` as an explicit source for `torch` in `pyproject.toml`. This prevents `uv` from pulling down massive GPU/CUDA binaries (~8GB), producing compact ML containers.
2. **Sequential Remote Builds**: Executed remote builds sequentially to prevent BuildKit socket resets due to concurrent compilation IO overhead on the VM.
3. **Symbol Resolution Self-Healing**: Fixed a startup race condition where database bootstrapping finished but the resolved symbols cache returned empty due to stale caching, causing `ValueError` during index pre-building. The cache now auto-refreshes if a lookup yields zero matches.

## Next Steps
- Monitor remote server resource usage under active predictions.
- Verify frontend prediction dashboard rendering.
