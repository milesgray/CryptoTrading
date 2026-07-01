#!/bin/bash

GITHUB_USERNAME=$1
GITHUB_PAT=$2

 1. Serve Service
echo "Building and pushing Serve Service..."
docker build -t crypto-trading-serve:latest -f Dockerfile.serve .
docker tag crypto-trading-serve:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/serve:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/serve:latest

# 2. Retrieval Service
echo "Building and pushing Retrieval Service..."
docker build -t crypto-trading-retrieval:latest -f Dockerfile.retrieval .
docker tag crypto-trading-retrieval:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/retrieval:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/retrieval:latest

# 3. Record Service
echo "Building and pushing Record Service..."
docker build -t crypto-trading-record:latest -f Dockerfile.record .
docker tag crypto-trading-record:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/record:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/record:latest

# 4. Frontend Service
echo "Building and pushing Frontend Service..."
docker build -t crypto-trading-frontend:latest -f Dockerfile.frontend .
docker tag crypto-trading-frontend:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/frontend:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/frontend:latest

# 5. Embed Service
echo "Building and pushing Embed Service..."
docker build -t crypto-trading-embed:latest -f services/embed/Dockerfile .
docker tag crypto-trading-embed:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/embed:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/embed:latest

# 6. Jepa Service
echo "Building and pushing Jepa Service..."
docker build -t crypto-trading-jepa:latest -f services/jepa/Dockerfile .
docker tag crypto-trading-jepa:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/jepa:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/jepa:latest

# 7. Pressure Service
echo "Building and pushing Pressure Service..."
docker build -t crypto-trading-pressure:latest -f services/pressure/Dockerfile .
docker tag crypto-trading-pressure:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/pressure:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/pressure:latest

# 8. Predict Service
echo "Building and pushing Predict Service..."
docker build -t crypto-trading-predict:latest -f services/predict/Dockerfile .
docker tag crypto-trading-predict:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/predict:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/predict:latest

# 9. Sentiment Service
echo "Building and pushing Sentiment Service..."
docker build -t crypto-trading-sentiment:latest -f services/sentiment/Dockerfile .
docker tag crypto-trading-sentiment:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/sentiment:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/sentiment:latest

# 10. Train Service
echo "Building and pushing Train Service..."
docker build -t crypto-trading-train:latest -f services/train/Dockerfile .
docker tag crypto-trading-train:latest ghcr.io/$GITHUB_USERNAME/crypto-trading/train:latest
docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/train:latest#
