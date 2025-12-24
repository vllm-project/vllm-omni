#!/bin/bash

# This script build the Ascend NPU docker image and run the offline inference inside the container.
# It serves a sanity check for compilation and basic model usage.
set -ex

# Base ubuntu image with basic ascend development libraries and python installed
VLLM_OMNI_REPO="https://github.com/vllm-project/vllm-omni.git"
BASE_IMAGE_NAME="quay.nju.edu.cn/ascend/vllm-ascend:v0.11.0rc2"
image_name="npu/vllm-omni-ci:${BUILDKITE_COMMIT}_${EPOCHSECONDS}"
# image_name="npu/vllm-ci:${BUILDKITE_COMMIT}_${EPOCHSECONDS}"
container_name="npu_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards
agent_idx=$(echo "${BUILDKITE_AGENT_NAME}" | awk -F'-' '{print $(NF-1)}')
echo "agent_idx: ${agent_idx}"
builder_name="cachebuilder${agent_idx}"
builder_cache_dir="/mnt/docker-cache${agent_idx}"
mkdir -p ${builder_cache_dir}

# Try building the docker image
cat <<EOF | DOCKER_BUILDKIT=1 docker build \
    --add-host pypi-cache:${PYPI_CACHE_HOST} \
    --builder ${builder_name} --cache-from type=local,src=${builder_cache_dir} \
                           --cache-to type=local,dest=${builder_cache_dir},mode=max \
    --build-arg BUILDKITE_PULL_REQUEST="${BUILDKITE_PULL_REQUEST}" \
    --build-arg BUILDKITE_PULL_REQUEST_REPO="${BUILDKITE_PULL_REQUEST_REPO}" \
    --progress=plain --load -t ${image_name} -f - .
FROM ${BASE_IMAGE_NAME}

# Define environments
ENV DEBIAN_FRONTEND=noninteractive

RUN pip config set global.index-url http://pypi-cache:${PYPI_CACHE_PORT}/pypi/simple && \
    pip config set global.trusted-host pypi-cache && \
    apt-get update -y && \
    apt-get install -y python3-pip git vim wget net-tools gcc g++ cmake libnuma-dev && \
    rm -rf /var/cache/apt/* && \
    rm -rf /var/lib/apt/lists/*

# Install for pytest to make the docker build cache layer always valid
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install pytest>=6.0  pytest-cov modelscope

COPY . .

# Install vllm-omni
WORKDIR /workspace
ARG VLLM_OMNI_REPO=https://github.com/vllm-project/vllm-omni.git
ARG VLLM_OMNI_TAG=main
ARG BUILDKITE_PULL_REQUEST
ARG BUILDKITE_PULL_REQUEST_REPO
RUN git config --global url."https://gh-proxy.test.osinfra.cn/https://github.com/".insteadOf "https://github.com/" && \
    if [ "\$BUILDKITE_PULL_REQUEST" != "false" ] && [ -n "\$BUILDKITE_PULL_REQUEST" ]; then \
        echo "Cloning and checking out PR #\$BUILDKITE_PULL_REQUEST..." && \
        git clone \$VLLM_OMNI_REPO /workspace/vllm-omni && \
        cd /workspace/vllm-omni && \
        git fetch origin pull/\$BUILDKITE_PULL_REQUEST/head:pr-\$BUILDKITE_PULL_REQUEST && \
        git checkout pr-\$BUILDKITE_PULL_REQUEST; \
    else \
        echo "Not a PR build, using main branch" && \
        git clone --depth 1 \$VLLM_OMNI_REPO /workspace/vllm-omni; \
    fi

RUN --mount=type=cache,target=/root/.cache/pip \
    export PIP_EXTRA_INDEX_URL=https://mirrors.huaweicloud.com/ascend/repos/pypi && \
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    source /usr/local/Ascend/nnal/atb/set_env.sh && \
    export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/local/Ascend/ascend-toolkit/latest/`uname -i`-linux/devlib && \
    python3 -m pip install -v -e /workspace/vllm-omni/

ENV VLLM_WORKER_MULTIPROC_METHOD=spawn

WORKDIR /workspace/vllm-omni
CMD ["/bin/bash"]

EOF

# Setup cleanup
remove_docker_container() {
  docker rm -f "${container_name}" || true;
  docker image rm -f "${image_name}" || true;
  docker system prune -f || true;
}
trap remove_docker_container EXIT

# Generate corresponding --device args based on BUILDKITE_AGENT_NAME
# Ascend NPU BUILDKITE_AGENT_NAME format is {hostname}-{agent_idx}-{npu_card_num}cards, and agent_idx starts from 1.
#   e.g. atlas-a2-001-1-2cards means this is the 1-th agent on atlas-a2-001 host, and it has 2 NPU cards.
#   returns --device /dev/davinci0 --device /dev/davinci1
parse_and_gen_devices() {
    local input="$1"
    local index cards_num
    if [[ "$input" =~ ([0-9]+)-([0-9]+)cards$ ]]; then
        index="${BASH_REMATCH[1]}"
        cards_num="${BASH_REMATCH[2]}"
    else
        echo "parse error" >&2
        return 1
    fi

    local devices=""
    local i=0
    while (( i < cards_num )); do
        local dev_idx=$(((index - 1)*cards_num + i ))
        devices="$devices --device /dev/davinci${dev_idx}"
        ((i++))
    done

    # trim leading space
    devices="${devices#"${devices%%[![:space:]]*}"}"
    # Output devices: assigned to the caller variable
    printf '%s' "$devices"
}

devices=$(parse_and_gen_devices "${BUILDKITE_AGENT_NAME}") || exit 1

# Run the image and execute the Out-Of-Tree (OOT) platform interface test case on Ascend NPU hardware.
# This test checks whether the OOT platform interface is functioning properly in conjunction with
# the hardware plugin vllm-ascend.
hf_model_cache_dir=/mnt/hf_cache${agent_idx}
ms_model_cache_dir=/mnt/modelscope${agent_idx}
mkdir -p ${hf_model_cache_dir}
mkdir -p ${ms_model_cache_dir}
docker run \
    --init \
    ${devices} \
    --device /dev/davinci_manager \
    --device /dev/devmm_svm \
    --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/ \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
    -v /etc/ascend_install.info:/etc/ascend_install.info \
    -v ${hf_model_cache_dir}:/root/.cache/huggingface \
    -v ${ms_model_cache_dir}:/root/.cache/modelscope \
    --network host \
    --entrypoint="" \
    --name "${container_name}" \
    "${image_name}" \
    bash -c '
    set -e
    VLLM_USE_MODELSCOPE=True pytest -s -v tests/e2e/offline_inference/test_qwen2_5_omni.py
'
