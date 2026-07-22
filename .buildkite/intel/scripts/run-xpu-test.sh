#!/bin/bash

# This script build the XPU docker image and run the offline inference inside the container.
set -ex

omni_source_dir=$(git rev-parse --show-toplevel)

base_image_name="xpu/vllm-omni-ci-base:${VLLM_VERSION:?VLLM_VERSION must be set}"
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

if [ -z "$(docker images -q "${base_image_name}")" ]; then
    docker_build "${base_image_name}" --target vllm-base --build-arg "VLLM_VERSION=${VLLM_VERSION}"
fi

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
    pytest -v -s -m "core_model and xpu and B60"
    pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
    export VLLM_XPU_USE_SAMPLER_KERNEL=0    # NOTE: Remove this after vLLM v0.21.1 is merged. Fixes Qwen2-5 omni-expansion tests.
    pytest -v -s -m "advanced_model and xpu and B60"
'
