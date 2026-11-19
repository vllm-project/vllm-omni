#!/bin/bash
# vllm-omni Intel bootstrap
# Keeps the existing pipeline-intel.yml path and offers an opt-in diff-aware Ready path.

set -euo pipefail

source .buildkite/common/scripts/resolve_skip_ci.sh

upload_pipeline() {
    echo "--- 🛠 Preparing Intel pipeline"
    buildkite-agent pipeline upload .buildkite/intel/pipeline-intel.yml
}

gate_bootstrap_ci intel l2

if [[ "${INTEL_XPU_READY:-0}" == "1" ]]; then
    echo "--- 🧪 Preparing optional Intel XPU Ready bootstrap pipeline"
    python3 .buildkite/common/scripts/upload_pipeline.py \
    --upload .buildkite/intel/bootstrap-upload-steps.yml
else
    upload_pipeline
fi
