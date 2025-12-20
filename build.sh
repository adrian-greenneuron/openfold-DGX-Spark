#!/bin/bash
# =============================================================================
# OpenFold Docker Build Script for DGX Spark
# =============================================================================
# Optimized for NVIDIA DGX Spark (ARM64 / Grace Blackwell GB10)
# Uses BuildKit for faster builds with cache mounts
# =============================================================================

set -euo pipefail

# Configuration
IMAGE_NAME="${IMAGE_NAME:-openfold-spark}"
IMAGE_TAG="${IMAGE_TAG:-cuda13}"
DOCKERFILE="${DOCKERFILE:-Dockerfile}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print with color
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Show usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build OpenFold Docker image for DGX Spark.

Options:
    -n, --name NAME     Image name (default: openfold-spark)
    -t, --tag TAG       Image tag (default: cuda13)
    -f, --file FILE     Dockerfile path (default: Dockerfile)
    --no-cache          Build without cache
    --test              Run quick import test after build
    -h, --help          Show this help

Examples:
    $0                          # Build openfold-spark:cuda13
    $0 -t latest                # Build openfold-spark:latest
    $0 --no-cache --test        # Full rebuild with testing

EOF
    exit 0
}

# Parse arguments
NO_CACHE=""
RUN_TEST=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--name)
            IMAGE_NAME="$2"
            shift 2
            ;;
        -t|--tag)
            IMAGE_TAG="$2"
            shift 2
            ;;
        -f|--file)
            DOCKERFILE="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --test)
            RUN_TEST=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            error "Unknown option: $1"
            usage
            ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# Pre-flight checks
info "Pre-flight checks..."

# Check Docker
if ! command -v docker &> /dev/null; then
    error "Docker not found. Please install Docker."
    exit 1
fi

# Check Dockerfile exists
if [[ ! -f "$DOCKERFILE" ]]; then
    error "Dockerfile not found: $DOCKERFILE"
    exit 1
fi

# Check for NVIDIA GPU (optional warning)
if ! nvidia-smi &> /dev/null; then
    warn "NVIDIA GPU not detected. Build will work, but testing requires GPU."
fi

# Enable BuildKit
export DOCKER_BUILDKIT=1

# Show build configuration
echo ""
info "Build Configuration:"
echo "  Image: ${FULL_IMAGE}"
echo "  Dockerfile: ${DOCKERFILE}"
echo "  BuildKit: Enabled"
echo "  Cache: $([ -z "$NO_CACHE" ] && echo 'Enabled' || echo 'Disabled')"
echo ""

# Build
info "Starting Docker build..."
BUILD_START=$(date +%s)

docker build \
    ${NO_CACHE} \
    -t "${FULL_IMAGE}" \
    -f "${DOCKERFILE}" \
    . 2>&1 | tee /tmp/docker_build_$$.log

BUILD_STATUS=$?
BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))

if [[ $BUILD_STATUS -ne 0 ]]; then
    error "Build failed! Check log: /tmp/docker_build_$$.log"
    exit 1
fi

success "Build completed in ${BUILD_DURATION}s"

# Show image info
info "Image details:"
docker images "${FULL_IMAGE}" --format "  Size: {{.Size}}\n  Created: {{.CreatedSince}}"

# Optional: Run quick test
if [[ "$RUN_TEST" == true ]]; then
    echo ""
    info "Running import test..."
    
    docker run --rm --gpus all "${FULL_IMAGE}" python3 -c "
import torch
import openfold
import openmm
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'OpenMM: {openmm.version.short_version}')
print('All imports successful!')
" 2>&1 | grep -v "^=\|^Copyright\|^NVIDIA\|^NOTE:\|^GOVERNING\|^(found at"

    if [[ $? -eq 0 ]]; then
        success "Import test passed!"
    else
        warn "Import test may have issues. Check output above."
    fi
fi

echo ""
success "Build complete: ${FULL_IMAGE}"
echo ""
info "Run inference with:"
echo "  docker run --gpus all --ipc=host --shm-size=64g ${FULL_IMAGE} python3 run_pretrained_openfold.py ..."
