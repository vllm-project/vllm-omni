#!/bin/bash

set -euo pipefail

# Get release version, default to 1.0.0.dev for nightly/per-commit builds
RELEASE_VERSION=$(buildkite-agent meta-data get release-version 2>/dev/null | sed 's/^v//')
if [ -z "${RELEASE_VERSION}" ]; then
  RELEASE_VERSION="1.0.0.dev"
fi

buildkite-agent annotate --style 'info' --context 'release-workflow' << EOF

To download and upload the image:

\`\`\`
docker pull public.ecr.aws/q9t5s3a7/vllm-omni-release-repo:${BUILDKITE_COMMIT}
docker pull public.ecr.aws/q9t5s3a7/vllm-omni-release-repo:${BUILDKITE_COMMIT}-rocm

docker tag public.ecr.aws/q9t5s3a7/vllm-omni-release-repo:${BUILDKITE_COMMIT} vllm/vllm-omni:v${RELEASE_VERSION}
docker push vllm/vllm-omni:v${RELEASE_VERSION}

docker tag public.ecr.aws/q9t5s3a7/vllm-omni-release-repo:${BUILDKITE_COMMIT}-rocm vllm/vllm-omni-rocm:v${RELEASE_VERSION}
docker push vllm/vllm-omni-rocm:v${RELEASE_VERSION}
\`\`\`
EOF
