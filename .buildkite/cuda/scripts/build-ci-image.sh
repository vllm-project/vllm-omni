#!/bin/bash
set -euo pipefail

# Build and push the CI Docker image with registry-based layer caching.
#
# Usage: build-ci-image.sh
#
# Expects the following environment variables (set by Buildkite):
#   BUILDKITE_COMMIT          - commit SHA used as the image tag
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
REGISTRY="${ECR_NAMESPACE}/vllm-ci-test-repo"
REGION="us-east-1"
DOCKERFILE="docker/Dockerfile.ci"
BUILDER_NAME="vllm-omni-builder"

# Ensure that the env vars are actually set, otherwise exit early
if [ -z "${BUILDKITE_COMMIT:-}" ]; then
    echo "ERROR: BUILDKITE_COMMIT is not set"
    exit 1
fi
echo "BUILDKITE_COMMIT: ${BUILDKITE_COMMIT}"

# Authenticate to ECR Public
aws ecr-public get-login-password --region "$REGION" \
    | docker login --username AWS --password-stdin "$ECR_NAMESPACE"

# Compute the cache tag for the dependencies image as the hash of the contents.
# This is because dependencies don't change often, so most PRs will have the same hash.
#
# The Dockerfile is also included to avoid churn in the cached image based on the vLLM version.
DEP_FILES="pyproject.toml setup.py requirements/*.txt docker/Dockerfile.ci"
CACHE_KEY=$(cat ${DEP_FILES} | sha256sum | cut -c1-16)
CACHE_TAG="deps-cache-${CACHE_KEY}"
echo "Cache key: ${CACHE_TAG}"

# Set up buildx with docker-container driver; we need to do this
# since cache export is not supported for the default docker driver
# that is running in the CI.
docker buildx inspect $BUILDER_NAME >/dev/null 2>&1 \
|| docker buildx create --name $BUILDER_NAME --driver docker-container
docker buildx use $BUILDER_NAME

echo "Building image tag: ${REGISTRY}:${BUILDKITE_COMMIT}"

# Build + push the image to the registry, using the hash from the dependencies.
# Note that the caching here is layerwise, so we're just making sure we always
# can cache hit on everything but the last copy at the moment.
docker buildx build --push --progress=plain \
    --cache-from "type=registry,ref=${REGISTRY}:${CACHE_TAG}" \
    --cache-to "type=registry,ref=${REGISTRY}:${CACHE_TAG},mode=max,compression=zstd" \
    --file "$DOCKERFILE" \
    -t "${REGISTRY}:${BUILDKITE_COMMIT}" .
