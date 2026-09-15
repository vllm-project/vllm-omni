#!/bin/bash

# This script build the XPU docker image and run the offline inference inside the container.
set -ex

omni_source_dir=$(git rev-parse --show-toplevel)

: "${VLLM_VERSION:?VLLM_VERSION must be set}"
# Official upstream XPU image. It is published from vllm's own
# docker/Dockerfile.xpu, which is exactly what build_vllm_base() clones and
# builds below, so pulling it is equivalent to building the fallback base.
upstream_base_image="vllm/vllm-openai-xpu:${VLLM_VERSION}"
local_base_image="xpu/vllm-omni-ci-base:${VLLM_VERSION}"
image_name="xpu/vllm-omni-ci:${BUILDKITE_COMMIT:?BUILDKITE_COMMIT must be set}"
container_name="xpu_${BUILDKITE_COMMIT}_$(
    tr -dc A-Za-z0-9 </dev/urandom | head -c 10
    echo
)"

cd "${omni_source_dir}"

# The XPU base image is ~37GB; the default gzip layer exporter is single-threaded
# and takes ~16min to compress it. zstd is multi-threaded (uses all cores) at a
# similar ratio; set EXPORT_COMPRESSION=uncompressed to skip compression entirely
# for local-only images. This requires the containerd image store (buildx docker
# driver), which is the default here.
EXPORT_COMPRESSION="${EXPORT_COMPRESSION:-zstd}"
if [ "${EXPORT_COMPRESSION}" = "uncompressed" ]; then
    export_args=(--output "type=image,name={{IMAGE}},compression=uncompressed")
else
    export_args=(--output "type=image,name={{IMAGE}},compression=${EXPORT_COMPRESSION},compression-level=3,force-compression=true")
fi

docker_build() {
    # $1 = image name; remaining args passed through to docker build.
    local image="$1"
    shift
    local out=("${export_args[@]/'{{IMAGE}}'/${image}}")
    docker build "${out[@]}" "$@" -f docker/Dockerfile.xpu .
}

build_vllm_base() (
    local vllm_source_dir
    vllm_source_dir=$(mktemp -d)
    trap 'rm -rf "${vllm_source_dir}"' EXIT

    git clone --depth 1 --branch "${VLLM_VERSION}" \
        https://github.com/vllm-project/vllm "${vllm_source_dir}"

    local out=("${export_args[@]/'{{IMAGE}}'/${base_image_name}}")
    docker build "${out[@]}" \
        --target vllm-openai \
        -f "${vllm_source_dir}/docker/Dockerfile.xpu" \
        "${vllm_source_dir}"
)

# Resolve the vLLM base image, in priority order:
#   1. the published upstream image for VLLM_VERSION, pulled fresh (so moving
#      tags like nightly are refreshed rather than served stale from disk);
#   2. a local copy of that same image, when the registry is unreachable;
#   3. VLLM_BASE, if that image is already on disk;
#   4. otherwise build VLLM_BASE from the matching vLLM source tag (~30min).
# VLLM_BASE names the fallback base and defaults to ${local_base_image}; set it
# to reuse or produce a base under a different tag.
resolve_vllm_base() {
    if docker pull "${upstream_base_image}"; then
        base_image_name="${upstream_base_image}"
        return
    fi

    # Registry unreachable, but a previous run may have left the image behind.
    if [ -n "$(docker images -q "${upstream_base_image}")" ]; then
        echo "WARNING: could not pull ${upstream_base_image}; using local copy" >&2
        base_image_name="${upstream_base_image}"
        return
    fi

    echo "WARNING: ${upstream_base_image} unavailable remotely and locally" >&2
    base_image_name="${VLLM_BASE:-${local_base_image}}"
    if [ -z "$(docker images -q "${base_image_name}")" ]; then
        echo "WARNING: building ${base_image_name} from vLLM ${VLLM_VERSION} source" >&2
        build_vllm_base
    fi
}

resolve_vllm_base

# Try building the docker image
docker_build "${image_name}" --build-arg "VLLM_BASE=${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}"

# Setup cleanup
remove_docker_container() {
    docker rm -f "${container_name}" || true
    docker image rm -f "${image_name}" || true
    docker system prune -f || true
}
trap remove_docker_container EXIT

HF_CACHE="${HF_CACHE:-$(realpath ~)/.cache/huggingface}"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

time timeout -k 30 30m docker run \
    --device /dev/dri:/dev/dri \
    --net=host \
    --ipc=host \
    -v /dev/dri/by-path:/dev/dri/by-path \
    -v "${HF_CACHE}:${HF_MOUNT}" \
    --security-opt seccomp=unconfined \
    --entrypoint="" \
    -e VLLM_LOGGING_LEVEL \
    -e VLLM_OMNI_LOGGING_LEVEL \
    -e HF_TOKEN \
    -e ZE_AFFINITY_MASK \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    echo $ZE_AFFINITY_MASK
    pip install tblib==3.1.0
    cd /workspace/vllm-omni
    XPU_TEST_PATHS="tests/diffusion tests/dfx tests/e2e"
    pytest -v -s $XPU_TEST_PATHS -m "core_model and xpu and B60"
    pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
    pytest -v -s $XPU_TEST_PATHS -m "advanced_model and xpu and B60"
    pytest -v -s $XPU_TEST_PATHS -m "omni and xpu and B60"
'
