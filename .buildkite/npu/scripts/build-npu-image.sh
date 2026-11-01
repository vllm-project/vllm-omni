#!/usr/bin/env bash
set -ex

dockerfile=${1:?Dockerfile is required}
image_tag=${2:?Image tag is required}
device_suffix=${3:-}

image_name=${IMAGE_NAME:-vllm-omni-ci-npu}
image_registry=${IMAGE_REGISTRY:-swr.cn-southwest-2.myhuaweicloud.com/modelfoundry}
buildkitd_addr=${BUILDKITD_ADDR:-tcp://buildkitd-service.buildkitd:1234}

echo "--- Building and pushing NPU Test Image"
echo "Image: ${image_registry}/${image_name}:${image_tag}"
echo "Dockerfile: ${dockerfile}"

if ! command -v buildctl &> /dev/null; then
  mkdir -p /tmp/buildkit
  buildkit_version=v0.29.0
  wget -q \
    "https://github.com/moby/buildkit/releases/download/${buildkit_version}/buildkit-${buildkit_version}.linux-arm64.tar.gz" \
    -O /tmp/buildkit.tar.gz
  tar -xzf /tmp/buildkit.tar.gz -C /tmp/buildkit
  cp /tmp/buildkit/bin/buildctl /usr/local/bin/
fi

build_args=()
if [[ -n "${device_suffix}" ]]; then
  build_args+=(--opt "build-arg:VLLM_ASCEND_DEVICE_SUFFIX=${device_suffix}")
fi

export DOCKER_CONFIG=/home/user/.docker
buildctl \
  --addr="${buildkitd_addr}" \
  --tlscacert=/home/user/.docker/ca.pem \
  --tlscert=/home/user/.docker/cert.pem \
  --tlskey=/home/user/.docker/key.pem \
  build \
  --frontend dockerfile.v0 \
  --local context=. \
  --local dockerfile=./docker \
  --opt "filename=${dockerfile}" \
  "${build_args[@]}" \
  --secret id=dockerconfig,src=/home/user/.docker/config.json \
  --output "type=image,name=${image_registry}/${image_name}:${image_tag},push=true" \
  --progress=plain

echo "--- Image pushed successfully"
echo "${image_registry}/${image_name}:${image_tag}"
