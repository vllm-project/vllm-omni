#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
#
# Publish Docker images from ECR staging to DockerHub.
#
# Usage:
#   publish-release-images.sh                 # release mode (default)
#   publish-release-images.sh release
#   publish-release-images.sh nightly [TAG_VARIANT]
#
# Release mode tags: latest, v<version>
# Nightly mode tags: nightly (or <variant>-nightly), plus commit-pinned manifest
#
# TAG_VARIANT (nightly only) moves the floating tag to "<variant>-nightly"
# so cleanup can retain per-variant history independently.

set -euo pipefail

MODE="${1:-release}"
TAG_VARIANT="${2:-}"

DOCKERHUB_REPO="vllm/vllm-omni"
ECR_REPO="public.ecr.aws/q9t5s3a7/vllm-omni-release-repo"
COMMIT="$BUILDKITE_COMMIT"
ECR_SUFFIX=""
# Primary floating tags that each get their own per-arch DockerHub tags.
PRIMARY_TAGS=()
# Extra multi-arch manifest names that reuse the first primary tag's arch images.
EXTRA_MANIFEST_TAGS=()

case "${MODE}" in
  release)
    RELEASE_VERSION=$(buildkite-agent meta-data get release-version --default "" | sed 's/^v//')
    if [ -z "${RELEASE_VERSION}" ]; then
      echo "ERROR: release-version metadata not set"
      exit 1
    fi
    PRIMARY_TAGS=("latest" "v${RELEASE_VERSION}")
    ;;
  nightly)
    if [ -n "${TAG_VARIANT}" ]; then
      ECR_SUFFIX="-${TAG_VARIANT}"
      PRIMARY_TAG="${TAG_VARIANT}-nightly"
    else
      PRIMARY_TAG="nightly"
    fi
    # Per-arch tags only for the floating nightly name; commit-pinned is
    # a multi-arch manifest alias (matches upstream push-nightly-builds.sh).
    PRIMARY_TAGS=("${PRIMARY_TAG}")
    EXTRA_MANIFEST_TAGS=("${PRIMARY_TAG}-${COMMIT}")
    ;;
  *)
    echo "Usage: $0 {release|nightly} [TAG_VARIANT]"
    exit 2
    ;;
esac

echo "========================================"
echo "Publishing ${MODE} images to ${DOCKERHUB_REPO}"
echo "  Commit: ${COMMIT}"
echo "  ECR suffix: '${ECR_SUFFIX}'"
echo "  Primary tags: ${PRIMARY_TAGS[*]}"
if [ ${#EXTRA_MANIFEST_TAGS[@]} -gt 0 ]; then
  echo "  Extra manifests: ${EXTRA_MANIFEST_TAGS[*]}"
fi
echo "========================================"

# Login to ECR to pull staging images
aws ecr-public get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin public.ecr.aws/q9t5s3a7

SRC_X86="${ECR_REPO}:${COMMIT}-x86_64${ECR_SUFFIX}"
SRC_ARM="${ECR_REPO}:${COMMIT}-aarch64${ECR_SUFFIX}"

docker pull "${SRC_X86}"
docker pull "${SRC_ARM}"

publish_manifest() {
  local tag="$1"
  local arch_tag_base="$2"
  docker manifest rm "${DOCKERHUB_REPO}:${tag}" || true
  docker manifest create \
    "${DOCKERHUB_REPO}:${tag}" \
    "${DOCKERHUB_REPO}:${arch_tag_base}-x86_64" \
    "${DOCKERHUB_REPO}:${arch_tag_base}-aarch64"
  docker manifest push "${DOCKERHUB_REPO}:${tag}"
}

for tag in "${PRIMARY_TAGS[@]}"; do
  docker tag "${SRC_X86}" "${DOCKERHUB_REPO}:${tag}-x86_64"
  docker tag "${SRC_ARM}" "${DOCKERHUB_REPO}:${tag}-aarch64"
  docker push "${DOCKERHUB_REPO}:${tag}-x86_64"
  docker push "${DOCKERHUB_REPO}:${tag}-aarch64"
  publish_manifest "${tag}" "${tag}"
done

# Extra manifests reuse the first primary tag's per-arch images.
ARCH_BASE="${PRIMARY_TAGS[0]}"
for tag in "${EXTRA_MANIFEST_TAGS[@]}"; do
  publish_manifest "${tag}" "${ARCH_BASE}"
done

echo ""
echo "Successfully published ${MODE} images"
