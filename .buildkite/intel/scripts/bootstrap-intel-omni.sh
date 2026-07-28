#!/bin/bash
# vllm-omni Intel bootstrap
# Uses static pipeline-intel.yml for Intel XPU tests

set -euo pipefail

# minicpm-challenge branch: skip entire Intel/XPU CI (no pipeline upload => no image build / tests).
echo "--- Skipping Intel/XPU CI on this branch (including image build)"
exit 0

source .buildkite/common/scripts/resolve_skip_ci.sh

upload_pipeline() {
    echo "--- 🛠 Preparing Intel pipeline"
    buildkite-agent pipeline upload .buildkite/intel/pipeline-intel.yml
}

gate_bootstrap_ci intel l2

upload_pipeline
