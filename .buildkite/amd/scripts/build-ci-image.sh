#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

# Build and push the ROCm CI image with a registry-backed BuildKit cache. The
# AMD builders are ephemeral, so a local Docker cache cannot survive between
# jobs. Keeping the cache beside the runtime image also reuses the registry
# credentials already configured on the amd-cpu agents.

REGISTRY="${ROCM_CI_IMAGE_REGISTRY:-rocm/vllm-omni}"
DOCKERFILE="docker/Dockerfile.rocm"
BUILDER_NAME="vllm-omni-rocm-builder"
ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"

if [[ -z "${BUILDKITE_COMMIT:-}" ]]; then
    echo "ERROR: BUILDKITE_COMMIT is not set" >&2
    exit 1
fi

dependency_files=(
    "$DOCKERFILE"
    "${DOCKERFILE}.dockerignore"
    pyproject.toml
    setup.py
    tools/install_torchcodec_rocm.sh
)
while IFS= read -r dependency_file; do
    dependency_files+=("$dependency_file")
done < <(find requirements -maxdepth 1 -type f -name '*.txt' -print | LC_ALL=C sort)

for dependency_file in "${dependency_files[@]}"; do
    if [[ ! -f "$dependency_file" ]]; then
        echo "ERROR: cache-key input does not exist: $dependency_file" >&2
        exit 1
    fi
done

CACHE_KEY=$(
    {
        printf 'rocm-arch=%s\n' "$ROCM_ARCH"
        for dependency_file in "${dependency_files[@]}"; do
            printf 'path=%s\n' "$dependency_file"
            sha256sum "$dependency_file"
        done
    } | sha256sum | cut -c1-16
)
CACHE_REF="${REGISTRY}:rocm-deps-cache-${CACHE_KEY}"
IMAGE_REF="${REGISTRY}:${BUILDKITE_COMMIT}"

echo "ROCm dependency cache: ${CACHE_REF}"
echo "ROCm CI image: ${IMAGE_REF}"

# The docker-container driver supports exporting the full cache to a registry;
# the default docker driver on CI does not.
if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
    docker buildx create --name "$BUILDER_NAME" --driver docker-container
fi
docker buildx use "$BUILDER_NAME"
docker buildx inspect --bootstrap

docker buildx build --push --pull --progress=plain \
    --target test \
    --build-arg "ARG_PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
    --cache-from "type=registry,ref=${CACHE_REF}" \
    --cache-to "type=registry,ref=${CACHE_REF},mode=max,compression=zstd,ignore-error=true" \
    --file "$DOCKERFILE" \
    --tag "$IMAGE_REF" .
