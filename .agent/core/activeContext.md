# Active Context: Microservice Port Alignment & Service Client Health Methods

## Quick Reference
- **Feature**: Docker Compose Internal Service Port Alignment & Standardized Service Client Health Checks
- **Status**: Completed ✅

## Executive Summary
Resolved `[Errno 111] Connection refused` errors when `serve` calls `pressure` (`http://pressure:8382`) by setting explicit `PORT` environment variables and matching container-to-host port mappings across all microservices in `docker-compose.yml`. Standardized client implementations in `src/cryptotrading/client/` with `is_healthy()` and `check_health()` methods across all service clients.

## Tech Stack & Components
- **Docker Compose**: Set `PORT` environment variable and mapped `${PORT}:${PORT}` for `pressure` (8382), `retrieval` (8388), `embed` (8380), `price` (8387), `predict` (8381), `train` (8389), and `artifact` (8383).
- **Service Clients (`src/cryptotrading/client`)**: Added standardized health check and monitoring helpers across all microservice client classes.

## Key Files Modified
- [docker-compose.yml](file:///home/miles/Development/notebooks/CryptoTrading/docker-compose.yml): Added internal `PORT` variables and aligned port bindings.
- [src/cryptotrading/client/pressure/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/pressure/client.py): Added health check methods.
- [src/cryptotrading/client/price/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/price/client.py): Created PriceServerClient.
- [src/cryptotrading/client/embed/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/embed/client.py): Added health checks.
- [src/cryptotrading/client/retrieval/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/retrieval/client.py): Added health checks.
- [src/cryptotrading/client/predict/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/predict/client.py): Added health checks.
- [src/cryptotrading/client/train/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/train/client.py): Added health checks.
- [src/cryptotrading/client/sentiment/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/sentiment/client.py): Added health checks.
- [src/cryptotrading/client/serve/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/serve/client.py): Added health checks.
- [src/cryptotrading/client/trade/client.py](file:///home/miles/Development/notebooks/CryptoTrading/src/cryptotrading/client/trade/client.py): Added health checks.
