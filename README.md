# Crypto Trading Kubernetes Deployment

This repository contains Kubernetes manifests to deploy the Crypto Trading application. The deployment includes TimescaleDB, MongoDB, Portainer, and various microservices.

## Prerequisites

- Kubernetes cluster (Minikube, EKS, GKE, AKS, etc.)
- `kubectl` configured to access your cluster
- Docker (for building images)

## Deployment Steps

1. **Build Docker Images**
   ```bash
   docker build -t crypto-trading-serve:latest -f Dockerfile.serve .
   docker build -t crypto-trading-retrieval:latest -f Dockerfile.retrieval .
   docker build -t crypto-trading-record:latest -f Dockerfile.record .
   docker build -t crypto-trading-frontend:latest -f Dockerfile.frontend .
   docker build -t crypto-trading-embed:latest -f services/embed/Dockerfile .
   docker build -t crypto-trading-jepa:latest -f services/jepa/Dockerfile .
   docker build -t crypto-trading-pressure:latest -f services/pressure/Dockerfile .
   docker build -t crypto-trading-predict:latest -f services/predict/Dockerfile .
   docker build -t crypto-trading-sentiment:latest -f services/sentiment/Dockerfile .
   docker build -t crypto-trading-train:latest -f services/train/Dockerfile .
   ```

2. **Apply Kubernetes Manifests**
   ```bash
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
   ```

3. **Verify Deployment**
   ```bash
   kubectl get pods -n crypto-trading
   kubectl get services -n crypto-trading
   ```

## Accessing Services

- **Portainer**: Accessible at `http://localhost:9000`
- **Frontend**: Accessible at `http://localhost:8080`
- **Serve API**: Accessible at `http://localhost:8362`

## Cleanup

To remove the deployment:
```bash
kubectl delete -f k8s/ingress.yaml
kubectl delete -f k8s/train-service.yaml
kubectl delete -f k8s/train-deployment.yaml
kubectl delete -f k8s/sentiment-service.yaml
kubectl delete -f k8s/sentiment-deployment.yaml
kubectl delete -f k8s/predict-service.yaml
kubectl delete -f k8s/predict-deployment.yaml
kubectl delete -f k8s/pressure-service.yaml
kubectl delete -f k8s/pressure-deployment.yaml
kubectl delete -f k8s/jepa-service.yaml
kubectl delete -f k8s/jepa-deployment.yaml
kubectl delete -f k8s/embed-service.yaml
kubectl delete -f k8s/embed-deployment.yaml
kubectl delete -f k8s/frontend-service.yaml
kubectl delete -f k8s/frontend-deployment.yaml
kubectl delete -f k8s/record-service.yaml
kubectl delete -f k8s/record-deployment.yaml
kubectl delete -f k8s/retrieval-service.yaml
kubectl delete -f k8s/retrieval-deployment.yaml
kubectl delete -f k8s/serve-service.yaml
kubectl delete -f k8s/serve-deployment.yaml
kubectl delete -f k8s/portainer-service.yaml
kubectl delete -f k8s/portainer-deployment.yaml
kubectl delete -f k8s/mongo-service.yaml
kubectl delete -f k8s/mongo-deployment.yaml
kubectl delete -f k8s/timescaledb-service.yaml
kubectl delete -f k8s/timescaledb-deployment.yaml
kubectl delete -f k8s/pvc-portainer-data.yaml
kubectl delete -f k8s/pvc-mongo-data.yaml
kubectl delete -f k8s/pvc-pgdata.yaml
kubectl delete -f k8s/configmap.yaml
kubectl delete -f k8s/namespace.yaml
```

## Notes

- Ensure Docker images are built and available in your local registry or set `imagePullPolicy: Never` in the deployments.
- Adjust resource limits and requests in the deployments as needed for your cluster.
- For production, consider adding TLS certificates and proper domain names for the Ingress.
