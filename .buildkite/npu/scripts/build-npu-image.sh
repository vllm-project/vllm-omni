#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

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

# Frozen per-device vendor image bytes. A suffix cannot inherit the A2/B3
# digest: Docker resolves the digest even when a different tag is supplied.
case "${device_suffix}" in
  "") base_digest=sha256:4a666fd818058c26ce445fa890e31ba781d0d43c721631b0d43e2eb550fbef99 ;;
  -a3) base_digest=sha256:ec7cd7c98616d55500e21a17b63be3a99603d520087f41dc37b60ae6c75d0730 ;;
  -a5) base_digest=sha256:51dc15c552d8de7141f84815de9d8ff030bc97b413280db742640aa467d632db ;;
  -310p) base_digest=sha256:22f665099114310858daf90d072eb46ce97ebbbdce1f808879aee68a427de76f ;;
  *) echo "Unsupported Ascend device suffix: ${device_suffix}" >&2; exit 2 ;;
esac
build_args=(
  --opt "build-arg:VLLM_ASCEND_IMAGE=quay.io/atlas-ci/vllm-ascend"
  --opt "build-arg:VLLM_ASCEND_VERSION=v0.33.0"
  --opt "build-arg:VLLM_ASCEND_DEVICE_SUFFIX=${device_suffix}"
  --opt "build-arg:VLLM_ASCEND_DIGEST=${base_digest}"
)

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
