#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Ban direct torch.cuda.* usage outside an allowlist.

Modeled on vllm/tools/pre_commit/check_torch_cuda.py. Call sites should use
current_omni_platform / OmniPlatform instead of CUDA-only torch.cuda APIs.
"""

import sys

import regex as re

# Match `torch.cuda.xxx` device/memory/seed helpers that have portable Omni
# platform equivalents. CUDAGraph / is_available / stream capture are not in
# this list (same split as upstream vLLM).
_TORCH_CUDA_PATTERNS = [
    r"\btorch\.cuda\.(empty_cache|synchronize|device_count|current_device|memory_reserved|memory_allocated|max_memory_allocated|max_memory_reserved|reset_peak_memory_stats|memory_stats|mem_get_info|set_device)\b",
    # Match torch.cuda.device(...) including args. `device\()\b` misses
    # `device()` (no word char after `)`) and `device(0)`.
    r"\btorch\.cuda\.device\s*\(",
    r"\btorch\.cuda\.(manual_seed|manual_seed_all)\b",
    r"\bwith\storch\.cuda\.device\b",
    # Calls torch.cuda.{_is_compiled/_device_count_amdsmi/_device_count_nvml} internally
    r"\bcuda_device_count_stateless\(\)\b",
]

# Platform adapters must call native torch.cuda / torch.npu / torch.xpu APIs.
ALLOWED_PREFIXES = {
    "vllm_omni/platforms/",
}

# Grandfathered call sites. Remove a path after migrating it to
# current_omni_platform / OmniPlatform. Do not add new files without review.
ALLOWED_FILES = {
    "benchmarks/kernels/mot_linear_benchmarks.py",
    "examples/online_serving/qwen2_5_omni/gradio_demo.py",
    "examples/online_serving/qwen3_omni/gradio_demo.py",
    "tests/dfx/reliability/helpers.py",
    "tests/diffusion/distributed/test_cfg_parallel.py",
    "tests/diffusion/distributed/test_pipeline_parallel.py",
    "tests/e2e/accuracy/wan22_i2v/run_wan22_i2v_diffusers_cp.py",
    "tests/e2e/features/custom_pipeline/test_async_omni_collective_rpc.py",
    "tests/e2e/features/fullduplex/test_personaplex_elastic.py",
    "tests/entrypoints/test_omni_sleep_mode.py",
    "tools/configure_stage_memory.py",
    "vllm_omni/diffusion/models/bagel/pipeline_bagel.py",
    "vllm_omni/diffusion/models/lance/pipeline_lance.py",
    "vllm_omni/diffusion/models/magi_human/pipeline_magi_human.py",
    "vllm_omni/diffusion/models/nextstep_1_1/pipeline_nextstep_1_1.py",
    "vllm_omni/distributed/omni_connectors/connectors/mooncake_transfer_engine_connector.py",
    "vllm_omni/distributed/omni_connectors/connectors/mori_transfer_engine_connector.py",
    "vllm_omni/experimental/ar_diffusion/runner.py",
    "vllm_omni/model_executor/models/moss_tts/cuda_graph_streaming_decoder_wrapper.py",
    "vllm_omni/model_executor/models/moss_tts_nano/modeling_moss_tts_nano.py",
    "vllm_omni/model_executor/models/qwen3_tts/cuda_graph_decoder_wrapper.py",
    "vllm_omni/worker/gpu_ar_model_runner.py",
}

_GENERIC_TIP = (
    "Found torch.cuda API call. Use current_omni_platform / OmniPlatform "
    "(empty_cache, synchronize, get_device_count, current_device, "
    "get_device_memory, set_device) instead."
)
_SEED_TIP = (
    "Found CUDA manual-seed API call. Use "
    "torch.Generator(device=current_omni_platform.device_type).manual_seed(...) "
    "instead."
)


def _normalize_path(path: str) -> str:
    return path.replace("\\", "/")


def _is_allowed(path: str) -> bool:
    if any(path.startswith(prefix) for prefix in ALLOWED_PREFIXES):
        return True
    return path in ALLOWED_FILES


def scan_file(path: str) -> int:
    with open(path, encoding="utf-8") as f:
        content = f.read()
    for pattern in _TORCH_CUDA_PATTERNS:
        for match in re.finditer(pattern, content, re.MULTILINE):
            line_start = content.rfind("\n", 0, match.start()) + 1
            if "#" in content[line_start : match.start()]:
                continue
            line_num = content[: match.start() + 1].count("\n") + 1
            matched_text = match.group(0)
            tip = _SEED_TIP if "manual_seed" in matched_text else _GENERIC_TIP
            print(f"{path}:{line_num}: \033[91merror:\033[0m {tip}")
            return 1
    return 0


def main():
    returncode = 0
    for filename in sys.argv[1:]:
        path = _normalize_path(filename)
        if _is_allowed(path):
            continue
        returncode |= scan_file(path)
    return returncode


if __name__ == "__main__":
    sys.exit(main())
