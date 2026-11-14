#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

COMMIT="${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
ECR_NAMESPACE="public.ecr.aws/q9t5s3a7"
IMAGE_REPO="${ECR_NAMESPACE}/vllm-ci-test-repo"
IMAGE_TAG="${COMMIT}-intel-xpu-ready"
BUILDER_NAME="vllm-omni-xpu-ready-builder"

aws ecr-public get-login-password --region us-east-1 \
  | docker login --username AWS --password-stdin "${ECR_NAMESPACE}"

docker buildx inspect "${BUILDER_NAME}" >/dev/null 2>&1 \
  || docker buildx create --name "${BUILDER_NAME}" --driver docker-container
docker buildx use "${BUILDER_NAME}"

docker buildx build --push --progress=plain \
  --target vllm-omni-openai \
  --build-arg "VLLM_VERSION=${VLLM_VERSION:-v0.33.0}" \
  --file docker/Dockerfile.xpu \
  --tag "${IMAGE_REPO}:${IMAGE_TAG}" .