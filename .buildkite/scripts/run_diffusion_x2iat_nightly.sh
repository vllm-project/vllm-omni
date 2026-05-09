#!/usr/bin/env bash

set -euo pipefail

readonly H100_MARK_EXPR="full_model and diffusion and H100"
readonly H100_SINGLE_GPU_MARK_EXPR="${H100_MARK_EXPR} and not distributed_cuda"
readonly H100_MULTI_GPU_MARK_EXPR="${H100_MARK_EXPR} and distributed_cuda"
readonly H100_RUN_LEVEL="full_model"

run() {
  echo "+ $*"
  "$@"
}

run_h100_diffusion_shard() {
  local mark_expr="${1:?missing mark expression}"
  shift

  local -a cmd=(pytest -sv "$@")
  cmd+=(-m "$mark_expr" --run-level "$H100_RUN_LEVEL")
  run "${cmd[@]}"
}

run_perf_config() {
  local config_path="$1"
  run pytest -s -v tests/dfx/perf/scripts/run_diffusion_benchmark.py --test-config-file "$config_path"
}

main() {
  local shard="${1:?missing shard name}"

  case "$shard" in
    function-single-gpu)
      run_h100_diffusion_shard "$H100_SINGLE_GPU_MARK_EXPR" \
        tests/e2e/online_serving/test_qwen_image_expansion.py \
        tests/e2e/online_serving/test_qwen_image_edit_expansion.py \
        tests/e2e/online_serving/test_qwen_image_layered_expansion.py \
        tests/e2e/online_serving/test_longcat_image_expansion.py \
        tests/e2e/online_serving/test_longcat_image_edit_expansion.py \
        tests/e2e/online_serving/test_flux_2_dev_expansion.py \
        tests/e2e/online_serving/test_bagel_expansion.py
      ;;
    function-qwen-parallel)
      run_h100_diffusion_shard "$H100_MULTI_GPU_MARK_EXPR" \
        tests/e2e/online_serving/test_qwen_image_expansion.py \
        tests/e2e/online_serving/test_qwen_image_edit_expansion.py
      ;;
    function-parallel-features)
      run_h100_diffusion_shard "$H100_MULTI_GPU_MARK_EXPR" \
        tests/e2e/online_serving/test_qwen_image_layered_expansion.py \
        tests/e2e/online_serving/test_longcat_image_expansion.py \
        tests/e2e/online_serving/test_longcat_image_edit_expansion.py \
        tests/e2e/online_serving/test_flux_2_dev_expansion.py
      ;;
    function-bagel-multi-gpu)
      run_h100_diffusion_shard "$H100_MULTI_GPU_MARK_EXPR" \
        tests/e2e/online_serving/test_bagel_expansion.py
      ;;
    perf-qwen-image)
      run_perf_config tests/dfx/perf/tests/test_qwen_image_vllm_omni.json
      ;;
    perf-qwen-image-edit)
      run_perf_config tests/dfx/perf/tests/test_qwen_image_edit_vllm_omni.json
      ;;
    perf-qwen-image-edit-2509)
      run_perf_config tests/dfx/perf/tests/test_qwen_image_edit_2509_vllm_omni.json
      ;;
    perf-qwen-image-layered)
      run_perf_config tests/dfx/perf/tests/test_qwen_image_layered_vllm_omni.json
      ;;
    *)
      echo "Unknown diffusion nightly shard: $shard" >&2
      exit 1
      ;;
  esac
}

main "$@"
