# Active Context: Forecasting Training Pipeline & RAFT Support

## Quick Reference
- **Feature**: Forecasting Training Pipeline, Index-Aware Dataloaders, and RAFT Model Support
- **Branch**: `feature/forecast-training-pipeline`
- **Status**: Completed & Verified ✅

## Executive Summary
Implement and integrate the training side of the timeseries forecasting engine in the `CryptoTrading` quantitative trading framework. Resolve package-level import pathways, implement index-aware timeseries datasets to yield absolute sample indices, and extend the training experiment runners to support the shape-similarity Retrieval-Augmented Forecasting Transformer (RAFT) model. Verify the entire pipeline with a comprehensive integration test suite.

## Key Accomplishments
- **Index-Aware Timeseries Dataset**: Updated `DataFramePriceForecastDataset` to yield 5-tuples containing absolute sample indices, enabling retrieval-augmented models to index their historical search databases correctly.
- **Package Import Resolutions**: Resolved all Python 3 relative/absolute import crashes and case-sensitive naming mismatches across the `cryptotrading.predict` submodules.
- **Flexible Experiment Runners**: Extended `ForecastExp` and `MovementExp` to support the custom training steps required by RAFT (pre-computation database generation and index-based forward signatures).
- **Dynamic Mode Selection**: Resolved an indexing out-of-bounds error during validation and testing loops by dynamically determining the RAFT forward pass mode (`'train'`, `'valid'`, `'test'`) based on the dataset's `set_type` attribute.
- **Resilient Movement Training**: Corrected a training loop bug in `MovementExp` where `self.test` was incorrectly called instead of `self.vali` at epoch ends.
- **Comprehensive Verification**: Developed and ran a robust test suite (`tests/test_training_pipeline.py`) verifying index-aware dataloaders, standard model training updates, and RAFT shape-similarity retrieval training. The full repository test suite passes with 100% success (8 passed).

## Next Objectives
- Implement historical database compression and retention policies on TimescaleDB.
- Benchmark training throughput and retrieval speed under simulated high-frequency workloads.
- Run database performance and latency load tests under simulated high-frequency updates.
- Integrate pgvector HNSW index queries into the forecasting logic.
