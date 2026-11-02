# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Warm the fused-MoE Triton kernel for prefill-sized batches at startup.

vLLM's startup warmup only runs the model at the CUDA-graph capture sizes
(e.g. ``[1, 2, 4, 5, 8]`` for the Qwen3-Omni thinker) and at the profile size
(``max_num_batched_tokens``). ``fused_moe_kernel`` picks its Triton launch
config from the batch size M: from a tuned config file keyed by M (the
Qwen3-Omni talker on H100 has one), or from ``get_default_config``'s M
thresholds when no file exists (the thinker on H100). Either way, prefills
between those two extremes land on configs that were never compiled, so the
first request of each size range JIT-compiles the kernel mid-request.

vLLM's JIT monitor logs this with ``warning_once``, so only the first such
compile per process is visible::

    Triton kernel JIT compilation during inference: fused_moe_kernel

With every compile logged, a fresh server compiles new configs on each of
the three turns of ``test_livekit_client_vad_replay``, in both the thinker
and the talker. On a cold Triton cache those compiles can push a realtime
turn past a client deadline (vllm-project/vllm-omni#9069).

This module runs one eager dummy forward per size in
``MOE_TUNING_BATCH_SIZES``, plus an odd-sized neighbour, before the JIT
monitor activates. The list matches the tuned-config keys and also straddles
every ``get_default_config`` threshold up to 4096. The odd neighbour matters
because Triton specializes integer arguments on divisibility by 16:
``fused_moe_kernel`` receives ``num_valid_tokens = M * top_k``, so even and odd
``M`` compile to different kernels for the same config.

Set ``VLLM_OMNI_MOE_PREFILL_WARMUP=0`` to skip it.
"""

from __future__ import annotations

import os
import time
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

# The batch sizes vLLM's MoE tuning configs are keyed by
# (benchmarks/kernels/benchmark_moe.py). ``try_get_optimal_moe_config`` maps a
# batch of M tokens to the nearest key, so one run per key covers its bucket.
# The same list straddles get_default_config's M thresholds (32, 64, 96, 128,
# 512), so it also covers models without a tuned config file.
MOE_TUNING_BATCH_SIZES: tuple[int, ...] = (
    1,
    2,
    4,
    8,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    256,
    512,
    1024,
    1536,
    2048,
    3072,
    4096,
)

_ENV_FLAG = "VLLM_OMNI_MOE_PREFILL_WARMUP"


def moe_prefill_warmup_enabled() -> bool:
    return os.environ.get(_ENV_FLAG, "1").strip().lower() not in {"0", "false", "no", "off"}


def moe_prefill_warmup_sizes(max_cudagraph_size: int, max_num_tokens: int) -> list[int]:
    """Return the eager batch sizes to warm.

    Sizes at or below ``max_cudagraph_size`` are already exercised by CUDA
    graph capture, and sizes above ``max_num_tokens`` can never be scheduled.
    """
    sizes: set[int] = set()
    for bucket in MOE_TUNING_BATCH_SIZES:
        for size in (bucket, bucket + 1):
            if max_cudagraph_size < size <= max_num_tokens:
                sizes.add(size)
    return sorted(sizes)


def model_has_fused_moe(model: Any) -> bool | None:
    """Return whether ``model`` has fused-MoE layers, or ``None`` if that cannot be checked."""
    # vLLM's fused-MoE layers are ``vllm.model_executor.layers.fused_moe.MoERunner``
    # modules. If vLLM renames or moves that symbol, warn instead of silently
    # skipping the warmup.
    try:
        from vllm.model_executor.layers.fused_moe import MoERunner
    except ImportError:
        logger.warning(
            "MoE prefill warmup could not check this model for fused-MoE layers: "
            "vllm.model_executor.layers.fused_moe.MoERunner is not importable. If the model "
            "has MoE layers, fused_moe_kernel configs may JIT-compile on the first requests."
        )
        return None
    modules = getattr(model, "modules", None)
    if modules is None:
        return False
    return any(isinstance(module, MoERunner) for module in modules())


def warm_up_moe_prefill_kernels(worker: Any) -> list[int]:
    """Run eager dummy forwards that compile every prefill fused-MoE config.

    Must run before vLLM activates the JIT monitor at the end of
    ``compile_or_warm_up_model``. Returns the sizes that were run.
    """
    if not moe_prefill_warmup_enabled():
        logger.info("Skipping MoE prefill warmup (%s=%s).", _ENV_FLAG, os.environ.get(_ENV_FLAG))
        return []
    if getattr(worker, "use_v2_model_runner", False):
        logger.info("Skipping MoE prefill warmup: not wired for the v2 model runner.")
        return []

    model_runner = worker.model_runner
    has_fused_moe = model_has_fused_moe(model_runner.get_model())
    if has_fused_moe is None:
        # model_has_fused_moe already warned that it could not check.
        return []
    if not has_fused_moe:
        logger.debug("Skipping MoE prefill warmup: the model has no fused-MoE layers.")
        return []

    compilation_config = worker.vllm_config.compilation_config
    capture_sizes = list(compilation_config.cudagraph_capture_sizes or [])
    max_cudagraph_size = max(capture_sizes, default=0)
    max_num_tokens = int(worker.scheduler_config.max_num_batched_tokens)
    sizes = moe_prefill_warmup_sizes(max_cudagraph_size, max_num_tokens)
    if not sizes:
        logger.info(
            "Skipping MoE prefill warmup: CUDA graph capture (up to %d tokens) already covers every "
            "MoE batch size up to max_num_batched_tokens=%d.",
            max_cudagraph_size,
            max_num_tokens,
        )
        return []

    start = time.perf_counter()
    for size in sizes:
        model_runner._dummy_run(size, skip_eplb=True, remove_lora=False)
    logger.info(
        "MoE prefill warmup ran %d batch sizes (%d..%d) in %.2fs.",
        len(sizes),
        sizes[0],
        sizes[-1],
        time.perf_counter() - start,
    )
    return sizes
