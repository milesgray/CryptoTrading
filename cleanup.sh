#!/bin/bash

# Cleanup Kubernetes deployment
cleanup_k8s() {
    echo "Cleaning up Kubernetes deployment..."
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
    echo "Kubernetes deployment cleaned up successfully."
}

# Main script execution
main() {
    cleanup_k8s
}

main
