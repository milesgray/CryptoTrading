# Deployment to American Cloud Services



### STEP A: Install
Ensure helm and kubectl are installed

### Step B: Configure Kubernetes Access
Ensure `kubectl` is pointing to the ACKS cluster by setting the `KUBECONFIG` environment variable:
```bash
export KUBECONFIG=kube.conf
kubectl cluster-info
```

### Step C: Apply ConfigMaps & Secrets
Create the `goldenage` namespace and configure application configuration parameters and credentials:
```bash
cd ../k8s

# Create namespace
kubectl apply -f namespace.yaml

# Apply ConfigMaps
kubectl apply -f configmap.yaml

# Create Application Secret
export $(grep -v '^#' ../.env | xargs) && \
  kubectl create secret generic app-secrets \
  --namespace=crypto-trading \
  --from-literal=JWT_SECRET="$JWT_SECRET" 

export KUBECONFIG=kube.conf
export $(grep -v '^#' ../.env | xargs) && \
kubectl create secret docker-registry registry-credentials \
  --namespace=crypto-trading \
  --docker-server=ghcr.io \
  --docker-username="$GITHUB_USERNAME" \
  --docker-password="$GITHUB_PAT"
```

### Step D: Expose the Cluster (Helm Controllers)
Install the ingress controller and cert-manager:
```bash
# Ingress Controller
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx --namespace default --set controller.publishService.enabled=true

# Cert-Manager
helm repo add jetstack https://charts.jetstack.io
helm install cert-manager jetstack/cert-manager --namespace cert-manager --create-namespace --version v1.12.0 --set installCRDs=true
```

### Push Images to github

```bash
# 1. API Gateway Service
docker build -t ghcr.io/milesgray/goldenage-hub/gateway:latest -f backend/Dockerfile.gateway backend/
docker push ghcr.io/milesgray/goldenage-hub/gateway:latest

# 2. Celery Pipeline Worker
docker build -t ghcr.io/milesgray/goldenage-hub/pipeline:latest -f backend/Dockerfile.pipeline backend/
docker push ghcr.io/milesgray/goldenage-hub/pipeline:latest

# 3. Policy Service
docker build -t ghcr.io/milesgray/goldenage-hub/policy:latest -f backend/Dockerfile.policy .
docker push ghcr.io/milesgray/goldenage-hub/policy:latest

```

### Step E: Apply Services & Deployments
Deploy the microservice resources and network policies:
```bash
# Network Boundaries
kubectl apply -f networkpolicy.yaml

# Policy Service
kubectl apply -f policy-service.yaml
kubectl apply -f policy-deployment.yaml

# API Gateway Service
kubectl apply -f gateway-service.yaml
kubectl apply -f gateway-deployment.yaml

# Celery Ingestion pipeline
kubectl apply -f pipeline-deployment.yaml

# Cluster SSL Issuer & Ingress Routing
kubectl apply -f letsencrypt-issuer.yaml
kubectl apply -f ingress.yaml
```

### Step F: Verify Deployment Status
Verify that all pods have spun up successfully:
```bash
kubectl rollout status deployment/policy -n goldenage
kubectl rollout status deployment/gateway -n goldenage
kubectl rollout status deployment/pipeline -n goldenage
```
Once healthy, configure your domain's DNS settings to map to the LoadBalancer external IP:
```bash
kubectl get svc -n default ingress-nginx-controller
```
