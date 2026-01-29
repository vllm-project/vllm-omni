#!/bin/bash
# vllm-omni customized version
# Based on: vllm/.buildkite/scripts/hardware_ci/run-amd-test.sh
# Last synced: 2025-12-15
# Modifications: docker image name for vllm-omni

# This script runs test inside the corresponding ROCm docker container.
set -o pipefail

# Export Python path
export PYTHONPATH=".."

# Print ROCm version
echo "--- Confirming Clean Initial State"
while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

echo "--- ROCm info"
rocminfo

# cleanup older docker images
cleanup_docker() {
  # Get Docker's root directory
  docker_root=$(docker info -f '{{.DockerRootDir}}')
  if [ -z "$docker_root" ]; then
    echo "Failed to determine Docker root directory."
    exit 1
  fi
  echo "Docker root directory: $docker_root"
  # Check disk usage of the filesystem where Docker's root directory is located
  disk_usage=$(df "$docker_root" | tail -1 | awk '{print $5}' | sed 's/%//')
  # Define the threshold
  threshold=70
  if [ "$disk_usage" -gt "$threshold" ]; then
    echo "Disk usage is above $threshold%. Cleaning up Docker images and volumes..."
    # Remove dangling images (those that are not tagged and not used by any container)
    docker image prune -f
    # Remove unused volumes / force the system prune for old images as well.
    docker volume prune -f && docker system prune --force --filter "until=72h" --all
    echo "Docker images and volumes cleanup completed."
  else
    echo "Disk usage is below $threshold%. No cleanup needed."
  fi
}

# Call the cleanup docker function
cleanup_docker

echo "--- Resetting GPUs"

echo "reset" > /opt/amdgpu/etc/gpu_state

while true; do
        sleep 3
        if grep -q clean /opt/amdgpu/etc/gpu_state; then
                echo "GPUs state is \"clean\""
                break
        fi
done

echo "--- Pulling container"
image_name="public.ecr.aws/q9t5s3a7/vllm-ci-test-repo:${BUILDKITE_COMMIT}-rocm-omni"
container_name="rocm_${BUILDKITE_COMMIT}_$(tr -dc A-Za-z0-9 < /dev/urandom | head -c 10; echo)"

# Install AWS CLI to authenticate to ECR Public Gallery to get higher rate limit for pulling images
sudo apt-get update && sudo apt-get install -y awscli
# Use safe docker login helper to prevent race conditions
source "$(dirname "${BASH_SOURCE[0]}")/../docker_login_ecr_public.sh"
safe_docker_login_ecr_public
# Pull the container from ECR Public Gallery

docker pull "${image_name}"

remove_docker_container() {
   docker rm -f "${container_name}" || docker image rm -f "${image_name}" || true
}
trap remove_docker_container EXIT

echo "--- Running container"

HF_CACHE="$(realpath ~)/huggingface"
mkdir -p "${HF_CACHE}"
HF_MOUNT="/root/.cache/huggingface"

commands=$@
echo "Commands:$commands"

PARALLEL_JOB_COUNT=8
MYPYTHONPATH=".."

# Test that we're launching on the machine that has
# proper access to GPUs
render_gid=$(getent group render | cut -d: -f3)
if [[ -z "$render_gid" ]]; then
  echo "Error: 'render' group not found. This is required for GPU access." >&2
  exit 1
fi

# check if the command contains shard flag, we will run all shards in parallel because the host have 8 GPUs.
if [[ $commands == *"--shard-id="* ]]; then
  # assign job count as the number of shards used
  commands=$(echo "$commands" | sed -E "s/--num-shards[[:blank:]]*=[[:blank:]]*[0-9]*/--num-shards=${PARALLEL_JOB_COUNT} /g" | sed 's/ \\ / /g')
  for GPU in $(seq 0 $(($PARALLEL_JOB_COUNT-1))); do
    # assign shard-id for each shard
    commands_gpu=$(echo "$commands" | sed -E "s/--shard-id[[:blank:]]*=[[:blank:]]*[0-9]*/--shard-id=${GPU} /g" | sed 's/ \\ / /g')
    echo "Shard ${GPU} commands:$commands_gpu"
    echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
    docker run \
        --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
        --network=host \
        --shm-size=16gb \
        --group-add "$render_gid" \
        --rm \
        -e MIOPEN_DEBUG_CONV_DIRECT=0 \
        -e MIOPEN_DEBUG_CONV_GEMM=0 \
        -e VLLM_ROCM_USE_AITER=1 \
        -e HIP_VISIBLE_DEVICES="${GPU}" \
        -e HF_TOKEN \
        -e AWS_ACCESS_KEY_ID \
        -e AWS_SECRET_ACCESS_KEY \
        -v "${HF_CACHE}:${HF_MOUNT}" \
        -e "HF_HOME=${HF_MOUNT}" \
        -e "PYTHONPATH=${MYPYTHONPATH}" \
        --name "${container_name}_${GPU}" \
        "${image_name}" \
        /bin/bash -c "${commands_gpu}" \
        |& while read -r line; do echo ">>Shard $GPU: $line"; done &
    PIDS+=($!)
  done
  #wait for all processes to finish and collect exit codes
  for pid in "${PIDS[@]}"; do
    wait "${pid}"
    STATUS+=($?)
  done
  for st in "${STATUS[@]}"; do
    if [[ ${st} -ne 0 ]]; then
      echo "One of the processes failed with $st"
      exit "${st}"
    fi
  done
else
  echo "Render devices: $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES"
  docker run \
          --device /dev/kfd $BUILDKITE_AGENT_META_DATA_RENDER_DEVICES \
          --network=host \
          --shm-size=16gb \
          --group-add "$render_gid" \
          --rm \
          -e MIOPEN_DEBUG_CONV_DIRECT=0 \
          -e MIOPEN_DEBUG_CONV_GEMM=0 \
          -e VLLM_ROCM_USE_AITER=1 \
          -e HF_TOKEN \
          -e AWS_ACCESS_KEY_ID \
          -e AWS_SECRET_ACCESS_KEY \
          -v "${HF_CACHE}:${HF_MOUNT}" \
          -e "HF_HOME=${HF_MOUNT}" \
          -e "PYTHONPATH=${MYPYTHONPATH}" \
          --name "${container_name}" \
          "${image_name}" \
          /bin/bash -c "${commands}"
fi
