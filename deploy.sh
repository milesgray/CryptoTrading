#!/bin/bash
export $(grep -v '^#' ./.env | xargs)

namespace_secrets() {
        # Create namespace
    kubectl apply -f namespace.yaml

    # Apply ConfigMaps
    kubectl apply -f configmap.yaml

    # Create Application Secret
    export $(grep -v '^#' ./.env | xargs) && \
    kubectl create secret generic app-secrets \
        --namespace=crypto-trading \
        --from-literal=AMERICAN_CLOUD_CLIENT_ID="$AMERICAN_CLOUD_CLIENT_ID" \
        --from-literal=AMERICAN_CLOUD_API_SECRET="$AMERICAN_CLOUD_API_SECRET" \
        --from-literal=AMERICAN_CLOUD_ACCESS_KEY="$AMERICAN_CLOUD_ACCESS_KEY" \
        --from-literal=AMERICAN_CLOUD_SECRET_KEY="$AMERICAN_CLOUD_SECRET_KEY"

    export KUBECONFIG=kube.conf
    export $(grep -v '^#' ./.env | xargs) && \
    kubectl create secret docker-registry registry-credentials \
        --namespace=crypto-trading \
        --docker-server=ghcr.io \
        --docker-username="$GITHUB_USERNAME" \
        --docker-password="$GITHUB_PAT"
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    # 1. Serve Service
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
    docker push ghcr.io/$GITHUB_USERNAME/crypto-trading/train:latest

    echo "Docker images built successfully."
}

# Deploy Kubernetes manifests
deploy_k8s() {
    echo "Deploying Kubernetes manifests..."
    kubectl apply -f k8s/namespace.yaml
    kubectl apply -f k8s/configmap.yaml
    kubectl apply -f k8s/pvc-pgdata.yaml
    kubectl apply -f k8s/pvc-mongo-data.yaml
    kubectl apply -f k8s/pvc-portainer-data.yaml
    kubectl apply -f k8s/timescaledb-deployment.yaml
    kubectl apply -f k8s/timescaledb-service.yaml
    kubectl apply -f k8s/mongo-deployment.yaml
    kubectl apply -f k8s/mongo-service.yaml
    kubectl apply -f k8s/portainer-deployment.yaml
    kubectl apply -f k8s/portainer-service.yaml
    kubectl apply -f k8s/serve-deployment.yaml
    kubectl apply -f k8s/serve-service.yaml
    kubectl apply -f k8s/retrieval-deployment.yaml
    kubectl apply -f k8s/retrieval-service.yaml
    kubectl apply -f k8s/record-deployment.yaml
    kubectl apply -f k8s/record-service.yaml
    kubectl apply -f k8s/frontend-deployment.yaml
    kubectl apply -f k8s/frontend-service.yaml
    kubectl apply -f k8s/embed-deployment.yaml
    kubectl apply -f k8s/embed-service.yaml
    kubectl apply -f k8s/jepa-deployment.yaml
    kubectl apply -f k8s/jepa-service.yaml
    kubectl apply -f k8s/pressure-deployment.yaml
    kubectl apply -f k8s/pressure-service.yaml
    kubectl apply -f k8s/predict-deployment.yaml
    kubectl apply -f k8s/predict-service.yaml
    kubectl apply -f k8s/sentiment-deployment.yaml
    kubectl apply -f k8s/sentiment-service.yaml
    kubectl apply -f k8s/train-deployment.yaml
    kubectl apply -f k8s/train-service.yaml
    kubectl apply -f k8s/ingress.yaml
    echo "Kubernetes manifests applied successfully."
}

# Verify deployment
verify_deployment() {
    echo "Verifying deployment..."
    kubectl get pods -n crypto-trading
    kubectl get services -n crypto-trading
    echo "Deployment verified."
}

# Main script execution
main() {
    namespace_secrets
    build_images
    deploy_k8s
    verify_deployment
}

main
