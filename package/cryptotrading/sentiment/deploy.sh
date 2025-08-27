#!/bin/bash
# Deployment script for Crypto Sentiment Analyzer

set -e

echo "🚀 Deploying Crypto Sentiment Analyzer..."

# Check if .env file exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo "Please create .env file with your Twitter API credentials:"
    echo ""
    echo "TWITTER_BEARER_TOKEN=your_bearer_token_here"
    echo "TWITTER_API_KEY=your_api_key_here"
    echo "TWITTER_API_SECRET=your_api_secret_here"
    echo "TWITTER_ACCESS_TOKEN=your_access_token_here"
    echo "TWITTER_ACCESS_TOKEN_SECRET=your_access_token_secret_here"
    echo ""
    exit 1
fi

# Create required directories
echo "📁 Creating directories..."
mkdir -p logs config mongo-init

# Copy config file if it doesn't exist
if [ ! -f config/config.yaml ]; then
    echo "📋 Creating default config file..."
    cp config.yaml config/config.yaml
fi

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be healthy
echo "⏳ Waiting for services to be ready..."
sleep 30

# Check service health
echo "🔍 Checking service health..."
if curl -s http://localhost:8080/health > /dev/null; then
    echo "✅ Service is healthy!"
    echo ""
    echo "📊 Service URLs:"
    echo "   Health Check: http://localhost:8080/health"
    echo "   Status:       http://localhost:8080/status"
    echo "   Metrics:      http://localhost:8080/metrics"
    echo "   Config:       http://localhost:8080/config"
    echo ""
    echo "🗄️  Database:"
    echo "   MongoDB:      mongodb://localhost:27017/crypto_sentiment"
    echo "   Mongo Express: http://localhost:8081 (run with --profile debug)"
    echo ""
    echo "📋 To view logs:"
    echo "   docker-compose logs -f crypto-sentiment-analyzer"
    echo ""
    echo "🛑 To stop:"
    echo "   docker-compose down"
else
    echo "❌ Service health check failed!"
    echo "Check logs with: docker-compose logs crypto-sentiment-analyzer"
    exit 1
fi