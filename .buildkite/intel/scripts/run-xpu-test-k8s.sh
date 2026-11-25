#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

set -euo pipefail

test_suite="${1:-all}"
test_cards="${2:-1}"
test_paths=(tests/diffusion tests/dfx tests/e2e tests/platforms
            tests/model_executor tests/entrypoints tests/engine
            tests/worker tests/worker_v2)

case "${test_cards}" in
    1|2) ;;
    *) echo "Unsupported XPU card count: ${test_cards}" >&2; exit 2 ;;
esac

cd /vllm-omni
pip install tblib==3.1.0
python -c 'import sys; import torch; count = torch.xpu.device_count(); print(f"Visible XPU devices: {count}"); sys.exit(count < int(sys.argv[1]))' "${test_cards}"

test_marker="xpu and (B60 or B70) and cards_${test_cards}"
test_args=("${test_paths[@]}" --ignore tests/e2e/offline_inference/test_cosyvoice3_expansion.py)

case "${test_suite}" in
    all)
        pytest -v -s "${test_args[@]}" -m "(core_model or omni) and not advanced_model and not full_model and ${test_marker}" --run-level core_model
        if [ "${test_cards}" = 1 ]; then
            pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
        fi
        pytest -v -s "${test_args[@]}" -m "advanced_model and not full_model and ${test_marker}" --run-level advanced_model
        ;;
    core_model)
        if [ "${test_cards}" = 1 ]; then
            model_marker="core_model and not omni"
            xpu_paths=()
            while IFS= read -r path; do
                [[ -z "${path}" || "${path}" = \#* ]] && continue
                if [ -f "${path}" ]; then
                    xpu_paths+=("${path}")
                else
                    echo "WARNING: ${path} no longer exists" >&2
                fi
            done < .buildkite/intel/xpu-path-tests.txt
            if [ "${#xpu_paths[@]}" -eq 0 ]; then
                echo "No XPU dispatch test paths found" >&2
                exit 1
            fi
            pytest -v -s -n "${XPU_CPU_WORKERS:-8}" "${xpu_paths[@]}" \
                -m "core_model and cpu and not (cards_2 or cards_3 or cards_4 or cards_5 or cards_6 or cards_7 or cards_8)"
        else
            model_marker="(core_model or omni)"
        fi
        pytest -v -s "${test_args[@]}" -m "${model_marker} and not advanced_model and not full_model and ${test_marker}" --run-level core_model
        ;;
    advanced_model)
        if [ "${test_cards}" = 2 ]; then
            test_args+=(--ignore tests/entrypoints/test_omni_sleep_mode_xpu.py)
        fi
        pytest -v -s "${test_args[@]}" -m "advanced_model and not full_model and ${test_marker}" --run-level advanced_model
        ;;
    omni)
        pytest -v -s "${test_args[@]}" -m "omni and not advanced_model and not full_model and ${test_marker}" --run-level core_model
        ;;
    quantization)
        if [ "${test_cards}" != 1 ]; then
            echo "Quantization config tests require the single-card lane" >&2
            exit 2
        fi
        pytest -v -s tests/diffusion/quantization/test_mxfp8_config.py
        ;;
    *)
        echo "Unknown XPU_TEST_SUITE: ${test_suite}" >&2
        exit 2
        ;;
esac
