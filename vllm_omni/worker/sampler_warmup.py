# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Register the Triton top-k/top-p kernels for MRv2 startup warmup on CUDA.

On CUDA, the v2 model runner samples top-k/top-p rows with FlashInfer's fused
sampler, so vLLM registers its native top-k/top-p kernels for JIT warmup only
on ROCm, and its scheduler-driven sampler warmup (unseeded requests) never
reaches them. Requests with an explicit seed are not eligible for the fused
sampler and run those Triton kernels instead: ``_topk_topp_kernel`` for top-k
rows, plus the split top-p kernels for small batches with an active top-p.
Every Qwen3-Omni talker request is seeded (``seed: 42, top_k: 50`` in the
deploy config), so without this registration the first one JIT-compiles the
kernel mid-request.

The registration only adds compile keys; vLLM compiles them in
``kernel_warmup``, before the JIT monitor activates.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def register_top_k_top_p_warmup(worker: Any) -> bool:
    """Register vLLM's native top-k/top-p kernels with the v2 runner's JIT warmup registry.

    Must run before ``compile_or_warm_up_model`` reaches ``kernel_warmup``.
    Assumes the zero-argument ``register_top_k_top_p_warmups()`` of vLLM 0.33.0.
    Returns whether the kernels were registered.
    """
    if not getattr(worker, "use_v2_model_runner", False) or not current_platform.is_cuda():
        return False
    model_runner = worker.model_runner
    # Every sampling stage registers, not only stages whose deploy config sets a
    # seed: any request can carry its own ``seed`` with ``top_k``/``top_p``.
    if not getattr(model_runner, "is_last_pp_rank", True) or getattr(model_runner, "is_pooling_model", False):
        return False
    registry = getattr(model_runner, "jit_warmup_registry", None)
    if registry is None:
        return False

    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON:
        return False
    try:
        from vllm_omni.worker.v1_compat.sample_ops.topk_topp_sampler import (
            register_top_k_top_p_warmups,
        )
    except ImportError:
        logger.warning_once(
            "Skipping top-k/top-p sampler warmup: "
            "vllm_omni.worker.v1_compat.sample_ops.topk_topp_sampler.register_top_k_top_p_warmups is not importable. "
            "Seeded top-k/top-p requests may JIT-compile sampling kernels on the first requests."
        )
        return False

    try:
        with registry.activate():
            register_top_k_top_p_warmups()
    except TypeError as e:
        logger.warning_once(
            "Skipping top-k/top-p sampler warmup: register_top_k_top_p_warmups() no longer "
            "matches the expected zero-argument signature (%s). Seeded top-k/top-p requests "
            "may JIT-compile sampling kernels on the first requests.",
            str(e),
        )
        return False
    logger.debug("Registered the Triton top-k/top-p sampling kernels for JIT warmup.")
    return True
