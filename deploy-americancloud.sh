#!/bin/bash
# Golden Age Hub - American Cloud Local Deployment Script
# Automatically builds/pushes images and deploys manifests to ACKS without GitHub Actions.

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
ENV_FILE="infra/.env.prod"
KUBECONFIG_PATH=""
DRY_RUN=false
SKIP_BUILD=false

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --kubeconfig)
      KUBECONFIG_PATH="$2"
      shift 2
      ;;
    --env-file)
      ENV_FILE="$2"
      shift 2
      ;;
    --skip-build)
      SKIP_BUILD=true
      shift
      ;;
    --dry-run)
      DRY_RUN=true
      shift
      ;;
    --help)
      echo "Usage: $0 [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --kubeconfig PATH    Path to ACKS kube.conf (required if not exported in KUBECONFIG)"
      echo "  --env-file PATH      Path to environment file (default: infra/.env.prod)"
      echo "  --skip-build         Skip rebuilding and pushing Docker images"
      echo "  --dry-run            Show what would be executed without applying changes"
      echo "  --help               Show this help message"
      exit 0
      ;;
    *)
      echo -e "${RED}Unknown option: $1${NC}"
      exit 1
      ;;
  esac
done

# Resolve Kubeconfig
if [ -n "$KUBECONFIG_PATH" ]; then
  export KUBECONFIG="$KUBECONFIG_PATH"
fi

if [ -z "$KUBECONFIG" ]; then
  echo -e "${RED}Error: Kubeconfig not specified. Export KUBECONFIG or use --kubeconfig option.${NC}"
  exit 1
fi

# Ensure kubeconfig file exists
if [ ! -f "$KUBECONFIG" ]; then
  echo -e "${RED}Error: Kubeconfig file not found at '$KUBECONFIG'${NC}"
  exit 1
fi

# Load environment variables
if [ -f "$ENV_FILE" ]; then
  echo -e "${GREEN}Loading variables from $ENV_FILE...${NC}"
  export $(grep -v '^#' "$ENV_FILE" | xargs)
else
  echo -e "${RED}Error: Environment file '$ENV_FILE' not found.${NC}"
  exit 1
fi

# Validate essential env variables
REQUIRED_VARS=(
  "POSTGRES_PASSWORD"
  "AMERICAN_CLOUD_ACCESS_KEY"
  "AMERICAN_CLOUD_SECRET_KEY"
  "GITHUB_USERNAME"
  "GITHUB_PAT"
)

for var in "${REQUIRED_VARS[@]}"; do
  if [ -z "${!var}" ]; then
    echo -e "${RED}Error: Required environment variable $var is missing from $ENV_FILE.${NC}"
    exit 1
  fi
done

# Helper function to generate secrets
generate_secret() {
  openssl rand -base64 32 | tr -d "=+/" | cut -c1-32
}

# Auto-generate optional secrets if not set
if [ -z "$REDIS_PASSWORD" ]; then
  REDIS_PASSWORD=$(openssl rand -base64 12 | tr -d "=+/" | cut -c1-16)
  echo -e "${YELLOW}Warning: REDIS_PASSWORD not found in $ENV_FILE. Generated value: $REDIS_PASSWORD${NC}"
fi

if [ -z "$JWT_SECRET" ]; then
  JWT_SECRET=$(generate_secret)
  echo -e "${YELLOW}Warning: JWT_SECRET not found in $ENV_FILE. Generated dynamic value.${NC}"
fi

if [ -z "$API_KEY" ]; then
  API_KEY=$(openssl rand -hex 32)
  echo -e "${YELLOW}Warning: API_KEY not found in $ENV_FILE. Generated value: $API_KEY${NC}"
fi

if [ -z "$POLICY_ADMIN_TOKEN" ]; then
  POLICY_ADMIN_TOKEN="policy-admin-$(openssl rand -hex 8)"
  echo -e "${YELLOW}Warning: POLICY_ADMIN_TOKEN not found in $ENV_FILE. Generated value: $POLICY_ADMIN_TOKEN${NC}"
fi

if [ -z "$HEALTH_API_KEY" ]; then
  HEALTH_API_KEY="health-key-$(openssl rand -hex 8)"
  echo -e "${YELLOW}Warning: HEALTH_API_KEY not found in $ENV_FILE. Generated value: $HEALTH_API_KEY${NC}"
fi


# Define image properties
REGISTRY="ghcr.io"
REPO_OWNER=$(echo "$GITHUB_USERNAME" | tr '[:upper:]' '[:lower:]')
IMAGE_TAG="latest"

GATEWAY_IMAGE="crypto-trading-serve:latest"
PIPELINE_IMAGE="crypto-trading-retrieval:latest"
POLICY_IMAGE="crypto-trading-record:latest"
DASHBOARD_IMAGE="crypto-trading-frontend:latest"
SIMULATOR_DASHBOARD_IMAGE="crypto-trading-embed:latest"
SIMULATOR_IMAGE="crypto-trading-jepa:latest"

# Run command helper
run_command() {
  local cmd="$1"
  if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}[DRY RUN]${NC} $cmd"
  else
    echo -e "${GREEN}Running:${NC} $cmd"
    eval "$cmd"
  fi
}

echo -e "${GREEN}=== Golden Age Hub American Cloud Local Deployment ===${NC}"
echo "Using Kubeconfig: $KUBECONFIG"
echo "Target Namespace: goldenage"
echo ""

# 1. Build and push container images
if [ "$SKIP_BUILD" = false ]; then
  echo -e "${GREEN}Step 1: Building and pushing Docker images...${NC}"
  
  # Log in to registry
  run_command "echo \"$GITHUB_PAT\" | docker login $REGISTRY -u \"$GITHUB_USERNAME\" --password-stdin"

  # Gateway
  run_command "docker build -t $GATEWAY_IMAGE -f backend/Dockerfile.gateway backend/"
  run_command "docker push $GATEWAY_IMAGE"

  # Pipeline
  run_command "docker build -t $PIPELINE_IMAGE -f backend/Dockerfile.pipeline backend/"
  run_command "docker push $PIPELINE_IMAGE"

  # Policy
  run_command "docker build -t $POLICY_IMAGE -f backend/Dockerfile.policy ."
  run_command "docker push $POLICY_IMAGE"

  # Dashboard
  run_command "docker build -t $DASHBOARD_IMAGE -f frontend/dashboard/Dockerfile frontend/dashboard/"
  run_command "docker push $DASHBOARD_IMAGE"

  # Simulator Dashboard
  run_command "docker build -t $SIMULATOR_DASHBOARD_IMAGE -f frontend/simulator-dashboard/Dockerfile frontend/simulator-dashboard/"
  run_command "docker push $SIMULATOR_DASHBOARD_IMAGE"

  # Simulator Backend
  run_command "docker build -t $SIMULATOR_IMAGE -f backend/Dockerfile.simulator ."
  run_command "docker push $SIMULATOR_IMAGE"
else
  echo -e "${YELLOW}Skipping container image building${NC}"
fi

# 2. Connect to ACKS and set up namespace
echo -e "${GREEN}Step 2: Configuring Kubernetes namespace and secrets...${NC}"
run_command "kubectl create namespace goldenage --dry-run=client -o yaml | kubectl apply -f -"

# 3. Create or update secrets
run_command "kubectl create secret docker-registry registry-credentials \
  --namespace=goldenage \
  --docker-server=$REGISTRY \
  --docker-username=$GITHUB_USERNAME \
  --docker-password=$GITHUB_PAT \
  --dry-run=client -o yaml | kubectl apply -f -"

run_command "kubectl create secret generic app-secrets \
  --namespace=goldenage \
  --from-literal=POSTGRES_PASSWORD=\"$POSTGRES_PASSWORD\" \
  --from-literal=REDIS_PASSWORD=\"$REDIS_PASSWORD\" \
  --from-literal=AMERICAN_CLOUD_A2_ACCESS_KEY=\"$AMERICAN_CLOUD_ACCESS_KEY\" \
  --from-literal=AMERICAN_CLOUD_A2_SECRET_KEY=\"$AMERICAN_CLOUD_SECRET_KEY\" \
  --from-literal=JWT_SECRET=\"$JWT_SECRET\" \
  --from-literal=API_KEY=\"$API_KEY\" \
  --from-literal=POLICY_ADMIN_TOKEN=\"$POLICY_ADMIN_TOKEN\" \
  --from-literal=HEALTH_API_KEY=\"$HEALTH_API_KEY\" \
  --dry-run=client -o yaml | kubectl apply -f -"

# 4. Configure Ingress and cert-manager
echo -e "${GREEN}Step 3: Checking Helm controllers...${NC}"
if [ "$DRY_RUN" = false ]; then
  helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx || true
  helm repo add jetstack https://charts.jetstack.io || true
  helm repo add kedacore https://kedacore.github.io/charts || true
  helm repo update

  if ! helm status ingress-nginx -n default >/dev/null 2>&1; then
    helm install ingress-nginx ingress-nginx/ingress-nginx --namespace default --set controller.publishService.enabled=true
  fi

  if ! helm status cert-manager -n cert-manager >/dev/null 2>&1; then
    kubectl create namespace cert-manager --dry-run=client -o yaml | kubectl apply -f -
    helm install cert-manager jetstack/cert-manager --namespace cert-manager --version v1.12.0 --set installCRDs=true
  fi

  if ! helm status keda -n keda >/dev/null 2>&1; then
    kubectl create namespace keda --dry-run=client -o yaml | kubectl apply -f -
    helm install keda kedacore/keda --namespace keda
  fi
else
  echo -e "${YELLOW}[DRY RUN] Would check and install Ingress & Cert-Manager via Helm${NC}"
fi

# 5. Apply manifests
echo -e "${GREEN}Step 4: Preparing and applying manifests...${NC}"

# Define temporary dir for updated manifests
TMP_MANIFEST_DIR=$(mktemp -d)
cp -r k8s/* "$TMP_MANIFEST_DIR"/

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
    kubectl apply -f k8s/letsencrypt-issuer.yaml
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
    # build_images
    deploy_k8s
    verify_deployment
}

# Apply base configurations, policies and scaling
run_command "kubectl apply -f $TMP_MANIFEST_DIR/configmap.yaml"
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


# Cleanup tmp files
rm -rf "$TMP_MANIFEST_DIR"

# 6. Verify rollout
kubectl rollout status deployment/timescaledb -n crypto-trading --timeout=120s
kubectl rollout status deployment/mongo -n crypto-trading --timeout=120s
kubectl rollout status deployment/portainer -n crypto-trading --timeout=120s
kubectl rollout status deployment/serve -n crypto-trading --timeout=120s
kubectl rollout status deployment/retrieval -n crypto-trading --timeout=120s
kubectl rollout status deployment/record -n crypto-trading --timeout=120s
kubectl rollout status deployment/frontend -n crypto-trading --timeout=120s
kubectl rollout status deployment/embed -n crypto-trading --timeout=120s
kubectl rollout status deployment/jepa -n crypto-trading --timeout=120s
kubectl rollout status deployment/pressure -n crypto-trading --timeout=120s
kubectl rollout status deployment/predict -n crypto-trading --timeout=120s
kubectl rollout status deployment/sentiment -n crypto-trading --timeout=120s
kubectl rollout status deployment/train -n crypto-trading --timeout=120s
 deployment/serve -n crypto-trading --timeout=120s
kubectl rollout status deployment/retrieval -n crypto-trading --timeout=120s
kubectl rollout status deployment/record -n crypto-trading --timeout=120s
kubectl rollout status deployment/frontend -n crypto-trading --timeout=120s
kubectl rollout status deployment/embed -n crypto-trading --timeout=120s
kubectl rollout status deployment/jepa -n crypto-trading --timeout=120s
kubectl rollout status deployment/pressure -n crypto-trading --timeout=120s
kubectl rollout status deployment/predict -n crypto-trading --timeout=120s
kubectl rollout status deployment/sentiment -n crypto-trading --timeout=120s
kubectl rollout status deployment/train -n crypto-trading --timeout=120s
kubectl rollout status deployment/portainer -n crypto-trading --timeout=120s
kubectl rollout status deployment/serve -n crypto-trading --timeout=120s
kubectl rollout status deployment/retrieval -n crypto-trading --timeout=120s
kubectl rollout status deployment/record -n crypto-trading --timeout=120s
kubectl rollout status deployment/frontend -n crypto-trading --timeout=120s
kubectl rollout status deployment/embed -n crypto-trading --timeout=120s
kubectl rollout status deployment/jepa -n crypto-trading --timeout=120s
kubectl rollout status deployment/pressure -n crypto-trading --timeout=120s
kubectl rollout status deployment/predict -n crypto-trading --timeout=120s
kubectl rollout status deployment/sentiment -n crypto-trading --timeout=120s
kubectl rollout status deployment/train -n crypto-trading --timeout=120s
out=120s
kubectl rollout status deployment/portainer -n crypto-trading --timeout=120s
kubectl rollout status deployment/serve -n crypto-trading --timeout=120s
kubectl rollout status deployment/retrieval -n crypto-trading --timeout=120s
kubectl rollout status deployment/record -n crypto-trading --timeout=120s
kubectl rollout status deployment/frontend -n crypto-trading --timeout=120s
kubectl rollout status deployment/embed -n crypto-trading --timeout=120s
kubectl rollout status deployment/jepa -n crypto-trading --timeout=120s
kubectl rollout status deployment/pressure -n crypto-trading --timeout=120s
kubectl rollout status deployment/predict -n crypto-trading --timeout=120s
kubectl rollout status deployment/sentiment -n crypto-trading --timeout=120s
kubectl rollout status deployment/train -n crypto-trading --timeout=120s
