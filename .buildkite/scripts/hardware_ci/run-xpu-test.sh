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
if [ -z "$(docker images -q "${base_image_name}")" ]; then
    docker build --target vllm-base -t "${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -f docker/Dockerfile.xpu .
fi

# Try building the docker image
docker build --build-arg "VLLM_BASE=${base_image_name}" --build-arg "VLLM_VERSION=${VLLM_VERSION}" -t "${image_name}" -f docker/Dockerfile.xpu .

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
    pytest -v -s -m "advanced_model and xpu and B60"
'
