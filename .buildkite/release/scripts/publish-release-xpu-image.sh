#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Publish the single-platform XPU image from ECR staging to DockerHub.
#
# Usage:
#   publish-release-xpu-image.sh release
#   publish-release-xpu-image.sh nightly

set -euo pipefail

MODE="${1:-release}"
DOCKERHUB_REPO="vllm/vllm-omni-xpu"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-omni-release-repo"
COMMIT="${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
SRC_IMAGE="${ECR_REPO}:${COMMIT}-xpu"

case "${MODE}" in
  release)
    RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
    if [ -z "${RELEASE_VERSION}" ]; then
      echo "ERROR: release-version metadata not set"
      exit 1
    fi
    TAGS=("latest" "v${RELEASE_VERSION}")
    ;;
  nightly)
    TAGS=("nightly" "nightly-${COMMIT}")
    ;;
  *)
    echo "Usage: $0 {release|nightly}"
    exit 2
    ;;
esac

aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7
docker pull "${SRC_IMAGE}"

for tag in "${TAGS[@]}"; do
  docker tag "${SRC_IMAGE}" "${DOCKERHUB_REPO}:${tag}"
  docker push "${DOCKERHUB_REPO}:${tag}"
done

echo "Successfully published ${MODE} XPU image to ${DOCKERHUB_REPO}"
