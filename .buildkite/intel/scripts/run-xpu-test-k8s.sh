#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

test_suite="${1:-all}"
test_paths=(tests/diffusion tests/dfx tests/e2e tests/platforms)

cd /workspace/vllm-omni
pip install tblib==3.1.0

case "${test_suite}" in
    all)
        pytest -v -s "${test_paths[@]}" -m "core_model and xpu and B60"
        pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
        pytest -v -s "${test_paths[@]}" -m "advanced_model and xpu and B60"
        pytest -v -s "${test_paths[@]}" -m "omni and xpu and B60"
        ;;
    core_model)
        pytest -v -s "${test_paths[@]}" -m "core_model and xpu and B60"
        ;;
    advanced_model)
        pytest -v -s "${test_paths[@]}" -m "advanced_model and xpu and B60"
        ;;
    omni)
        pytest -v -s "${test_paths[@]}" -m "omni and xpu and B60"
        ;;
    quantization)
        pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
        ;;
    *)
        echo "Unknown XPU_TEST_SUITE: ${test_suite}" >&2
        exit 2
        ;;
esac
