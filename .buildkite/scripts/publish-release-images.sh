#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Publish release Docker images from ECR to DockerHub.
# Pulls per-arch images, tags with `latest` and versioned tags, pushes them,
# then creates and pushes a multi-arch manifest.

set -euo pipefail

RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  echo "ERROR: release-version metadata not set"
  exit 1
fi

COMMIT="$BUILDKITE_COMMIT"
DOCKERHUB_REPO="vllm/vllm-omni"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-omni-release-repo"

echo "========================================"
echo "Publishing release images v${RELEASE_VERSION}"
echo "  Commit: ${COMMIT}"
echo "========================================"

# Login to ECR to pull staging images
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

docker pull "${ECR_REPO}:${COMMIT}-x86_64"
docker pull "${ECR_REPO}:${COMMIT}-aarch64"

docker tag "${ECR_REPO}:${COMMIT}-x86_64" "${DOCKERHUB_REPO}:latest-x86_64"
docker tag "${ECR_REPO}:${COMMIT}-x86_64" "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-x86_64"
docker push "${DOCKERHUB_REPO}:latest-x86_64"
docker push "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-x86_64"

docker tag "${ECR_REPO}:${COMMIT}-aarch64" "${DOCKERHUB_REPO}:latest-aarch64"
docker tag "${ECR_REPO}:${COMMIT}-aarch64" "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-aarch64"
docker push "${DOCKERHUB_REPO}:latest-aarch64"
docker push "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-aarch64"

docker manifest rm "${DOCKERHUB_REPO}:latest" || true
docker manifest rm "${DOCKERHUB_REPO}:v${RELEASE_VERSION}" || true
docker manifest create "${DOCKERHUB_REPO}:latest" "${DOCKERHUB_REPO}:latest-x86_64" "${DOCKERHUB_REPO}:latest-aarch64"
docker manifest create "${DOCKERHUB_REPO}:v${RELEASE_VERSION}" "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-x86_64" "${DOCKERHUB_REPO}:v${RELEASE_VERSION}-aarch64"
docker manifest push "${DOCKERHUB_REPO}:latest"
docker manifest push "${DOCKERHUB_REPO}:v${RELEASE_VERSION}"

echo ""
echo "Successfully published release images for v${RELEASE_VERSION}"
