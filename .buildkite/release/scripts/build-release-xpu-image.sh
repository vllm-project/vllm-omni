#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Build the release XPU image and stage it in ECR for validation/publishing.

set -euo pipefail

COMMIT="${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-omni-release-repo"
IMAGE_TAG="${COMMIT}-xpu"

aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

DOCKER_BUILDKIT=1 docker build \
  --target vllm-omni-openai \
  --tag "${ECR_REPO}:${IMAGE_TAG}" \
  --progress plain \
  -f docker/Dockerfile.xpu .

docker push "${ECR_REPO}:${IMAGE_TAG}"
