#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

# Build and push the ROCm CI image with a registry-backed BuildKit cache. The
# AMD builders are ephemeral, so a local Docker cache cannot survive between
# jobs. Keeping the cache beside the runtime image also reuses the registry
# credentials already configured on the amd-cpu agents.

usage() {
    echo "Usage: $0 [cache|image|all]" >&2
}

if (($# > 1)); then
    usage
    exit 2
fi

PHASE="${1:-all}"
RUN_CACHE=0
RUN_IMAGE=0
case "$PHASE" in
    cache)
        RUN_CACHE=1
        ;;
    image)
        RUN_IMAGE=1
        ;;
    all)
        RUN_CACHE=1
        RUN_IMAGE=1
        ;;
    *)
        echo "ERROR: unsupported build phase: $PHASE" >&2
        usage
        exit 2
        ;;
esac

REGISTRY="${ROCM_CI_IMAGE_REGISTRY:-rocm/vllm-omni}"
DOCKERFILE="docker/Dockerfile.rocm"
BUILDER_NAME="vllm-omni-rocm-builder"
ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx942}"
CACHE_PROBE_TIMEOUT="${AMD_CACHE_PROBE_TIMEOUT:-1m}"
CACHE_EXPORT_TIMEOUT="${AMD_CACHE_EXPORT_TIMEOUT:-60m}"
IMAGE_PUSH_TIMEOUT="${AMD_IMAGE_PUSH_TIMEOUT:-15m}"
IMAGE_PUSH_ATTEMPTS="${AMD_IMAGE_PUSH_ATTEMPTS:-2}"

if ((RUN_IMAGE)) && [[ -z "${BUILDKITE_COMMIT:-}" ]]; then
    echo "ERROR: BUILDKITE_COMMIT is not set" >&2
    exit 1
fi

if ((RUN_IMAGE)) && ! [[ "$IMAGE_PUSH_ATTEMPTS" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: AMD_IMAGE_PUSH_ATTEMPTS must be a positive integer" >&2
    exit 1
fi

dependency_files=(
    "$DOCKERFILE"
    "${DOCKERFILE}.dockerignore"
    pyproject.toml
    setup.py
    tools/install_torchcodec_rocm.sh
    tools/build_gradio_pillow13.py
    tools/gradio_pillow13/*
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

echo "ROCm dependency cache: ${CACHE_REF}"
if ((RUN_IMAGE)); then
    IMAGE_REF="${REGISTRY}:${BUILDKITE_COMMIT}"
    echo "ROCm CI image: ${IMAGE_REF}"
fi

# The docker-container driver supports exporting the full cache to a registry;
# the default docker driver on CI does not.
prepare_builder() {
    if ! docker buildx inspect "$BUILDER_NAME" >/dev/null 2>&1; then
        docker buildx create --name "$BUILDER_NAME" --driver docker-container
    fi
    docker buildx use "$BUILDER_NAME"
    docker buildx inspect --bootstrap
}

run_with_retry() {
    local description=$1
    local timeout_value=$2
    shift 2

    local attempt
    local status=1
    for ((attempt = 1; attempt <= IMAGE_PUSH_ATTEMPTS; attempt++)); do
        echo "${description}: attempt ${attempt}/${IMAGE_PUSH_ATTEMPTS}"
        if timeout --signal=TERM --kill-after=1m "$timeout_value" "$@"; then
            return 0
        else
            status=$?
        fi

        if ((attempt < IMAGE_PUSH_ATTEMPTS)); then
            echo "WARNING: ${description} exited with status ${status}; retrying in 5 seconds" >&2
            sleep 5
        fi
    done

    echo "ERROR: ${description} failed after ${IMAGE_PUSH_ATTEMPTS} attempts (status ${status})" >&2
    return "$status"
}

seed_dependency_cache() {
    local cache_status

    # The cache reference is keyed only by dependency inputs. Once it exists,
    # exporting it again cannot improve cache reuse. Serialize this phase so
    # two jobs cannot both observe a miss and publish the same cache reference.
    if timeout --signal=TERM --kill-after=10s "$CACHE_PROBE_TIMEOUT" \
        docker buildx imagetools inspect "$CACHE_REF" >/dev/null 2>&1; then
        echo "ROCm dependency cache hit; skipping cache export"
    else
        echo "ROCm dependency cache miss; exporting dependency layers"
        if timeout --signal=TERM --kill-after=1m "$CACHE_EXPORT_TIMEOUT" \
            docker buildx build --pull --progress=plain \
                --target deps \
                --build-arg "ARG_PYTORCH_ROCM_ARCH=${ROCM_ARCH}" \
                --cache-from "type=registry,ref=${CACHE_REF}" \
                --cache-to "type=registry,ref=${CACHE_REF},mode=max,compression=zstd,ignore-error=true" \
                --file "$DOCKERFILE" .; then
            echo "ROCm dependency cache export completed"
        else
            cache_status=$?
            echo "WARNING: ROCm dependency cache export exited with status ${cache_status};" \
                "continuing; the image build may rebuild missing layers" >&2
        fi
    fi
}

push_commit_image() {
    local -a image_build=(
        docker buildx build --push --pull --progress=plain
        --target test
        --build-arg "ARG_PYTORCH_ROCM_ARCH=${ROCM_ARCH}"
        --cache-from "type=registry,ref=${CACHE_REF}"
        --file "$DOCKERFILE"
        --tag "$IMAGE_REF" .
    )
    run_with_retry "ROCm CI image push" "$IMAGE_PUSH_TIMEOUT" "${image_build[@]}"
}

prepare_builder
if ((RUN_CACHE)); then
    seed_dependency_cache
fi
if ((RUN_IMAGE)); then
    push_commit_image
fi
