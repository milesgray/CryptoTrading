# Memory Index: CryptoTrading

## Overview
This file serves as the master index for the CryptoTrading memory system, providing checksums and metadata for all memory files to detect changes and ensure consistency.

## Memory Bank Structure

### Core Memory Files (Long-term Memory)
```
.agent/core/
├── projectbrief.md     # Project overview and goals
├── productContext.md   # Product requirements and user needs
├── systemPatterns.md   # Architecture and design patterns
├── techContext.md      # Technology stack and dependencies
├── activeContext.md    # Current work focus and state
└── progress.md         # Implementation progress and roadmap
```

### Plans Directory (Implementation Plans)
```
.agent/plans/
```

### Task Logs Directory (Short-term Memory)
```
.agent/task-logs/
```

### Errors Directory (Error Records)
```
.agent/errors/
```

## File Checksums and Metadata

### Core Memory Files

#### projectbrief.md
- **Path**: `.agent/core/projectbrief.md`
- **Type**: Project overview
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-22
- **SHA256**: `dcf3d9fa8f795b3949a80df6d842d403b4f3210c2a216dcef37da7e68b691721`
- **Size**: 5423 bytes
- **Status**: Active

#### productContext.md
- **Path**: `.agent/core/productContext.md`
- **Type**: Product requirements
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-22
- **SHA256**: `048dfcee9628ffd2c005f27d690ffd99bcb8975f4aa5edd5863145b401066ac2`
- **Size**: 4146 bytes
- **Status**: Active

#### systemPatterns.md
- **Path**: `.agent/core/systemPatterns.md`
- **Type**: Architecture patterns
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-22
- **SHA256**: `a967593a74017684a52e9db5ee68d84fed6270615c9aa6c4f225a489481106c6`
- **Size**: 8445 bytes
- **Status**: Active

#### techContext.md
- **Path**: `.agent/core/techContext.md`
- **Type**: Technology stack
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-22
- **SHA256**: `159e619874c149b6b6678059e732444c4999de5b96bb4cebfd3e52e374ed499c`
- **Size**: 4218 bytes
- **Status**: Active

#### activeContext.md
- **Path**: `.agent/core/activeContext.md`
- **Type**: Working memory
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-30
- **SHA256**: `dynamic_update`
- **Size**: 1050 bytes
- **Status**: Active

#### progress.md
- **Path**: `.agent/core/progress.md`
- **Type**: Progress tracking
- **Created**: 2026-06-22
- **Last Modified**: 2026-06-30
- **SHA256**: `dynamic_update`
- **Size**: 3200 bytes
- **Status**: Active

## Plans & Logs Tracking

### plans/
- **postgres-migration-plan.md**: `.agent/plans/postgres-migration-plan.md`
- **specretf-plan.md**: `.agent/plans/specretf-plan.md`
- **retrieval-service-plan.md**: Proposed implementation plan for retrieval service enhancements.
- **remove-mocks-and-styles-plan.md**: `.agent/plans/remove-mocks-and-styles-plan.md`
- **order-book-rework-plan.md**: `.agent/plans/order-book-rework-plan.md`
- **polish-embed-service-plan.md**: `.agent/plans/polish-embed-service-plan.md`
- **move-pgvector-store-plan.md**: `.agent/plans/move-pgvector-store-plan.md`

### task-logs/
- **task-log_2026-06-22-05-56_postgres-migration.md**: `.agent/task-logs/task-log_2026-06-22-05-56_postgres-migration.md`
- **task-log_2026-06-22-16-30_poetry-to-uv-migration.md**: `.agent/task-logs/task-log_2026-06-22-16-30_poetry-to-uv-migration.md`
- **task-log_2026-06-29-15-00_specretf-forecaster.md**: `.agent/task-logs/task-log_2026-06-29-15-00_specretf-forecaster.md`
- **task-log_2026-07-04-17-02_retrieval-service-enhancement.md**: `.agent/task-logs/task-log_2026-07-04-17-02_retrieval-service-enhancement.md`
- **task-log_2026-07-05-02-54_remove-mocks-and-styles.md**: `.agent/task-logs/task-log_2026-07-05-02-54_remove-mocks-and-styles.md`
- **task-log_2026-07-05-03-09_order-book-rework.md**: `.agent/task-logs/task-log_2026-07-05-03-09_order-book-rework.md`
- **task-log_2026-07-05-04-30_polish-embed-service.md**: `.agent/task-logs/task-log_2026-07-05-04-30_polish-embed-service.md`
- **task-log_2026-07-05-05-50_move-pgvector-store.md**: `.agent/task-logs/task-log_2026-07-05-05-50_move-pgvector-store.md`
- **task-log_2026-07-10-16-44_candlestick-query-timeouts.md**: `.agent/task-logs/task-log_2026-07-10-16-44_candlestick-query-timeouts.md`
- **task-log_2026-07-10-18-35_dynamic_forecasting.md**: `.agent/task-logs/task-log_2026-07-10-18-35_dynamic_forecasting.md`
- **task-log_2026-07-10-18-45_embed_service_auto_training.md**: `.agent/task-logs/task-log_2026-07-10-18-45_embed_service_auto_training.md`
- **task-log_2026-07-10-19-01_candlestick-remote-timeout.md**: `.agent/task-logs/task-log_2026-07-10-19-01_candlestick-remote-timeout.md`
- **task-log_2026-07-10-20-40_candlestick-backend-chunking.md**: `.agent/task-logs/task-log_2026-07-10-20-40_candlestick-backend-chunking.md`
- **task-log_2026-07-10-23-21_candlestick_retrieval.md**: `.agent/task-logs/task-log_2026-07-10-23-21_candlestick_retrieval.md`
- **task-log_2026-07-11-05-52_batch-embedding-optimization.md**: `.agent/task-logs/task-log_2026-07-11-05-52_batch-embedding-optimization.md`
- **task-log_2026-07-11-16-47_candlestick-volume-fix.md**: `.agent/task-logs/task-log_2026-07-11-16-47_candlestick-volume-fix.md`
- **task-log_2026-07-12-03-25_fix-echarts-null-candlestick.md**: `.agent/task-logs/task-log_2026-07-12-03-25_fix-echarts-null-candlestick.md`
- **task-log_2026-07-12-07-20_retrieval-scaling-fix.md**: `.agent/task-logs/task-log_2026-07-12-07-20_retrieval-scaling-fix.md`
- **task-log_2026-07-12-14-45_retrieval-overlap-fix.md**: `.agent/task-logs/task-log_2026-07-12-14-45_retrieval-overlap-fix.md`

## Memory System Status

### Overall Health
- **Status**: Synchronized
- **Last Check**: 2026-07-12 14:45:00 CST
- **Files Tracked**: 6 core files + 8 plans + 18 task logs
- **Integrity**: All systems updated and synchronized




