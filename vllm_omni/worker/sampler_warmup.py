# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Register the Triton top-k/top-p kernel for MRv2 startup warmup on CUDA.

On CUDA, the v2 model runner samples top-k/top-p rows with FlashInfer's fused
sampler, so vLLM does not register ``_topk_topp_kernel`` for JIT warmup, and
its scheduler-driven sampler warmup (unseeded requests) never reaches it.
Requests with an explicit seed are not eligible for the fused sampler and run
that Triton kernel instead. Every Qwen3-Omni talker request does
(``seed: 42, top_k: 50`` in the deploy config), so without this registration
the first one JIT-compiles the kernel mid-request.

The registration only adds compile keys; vLLM compiles them in
``kernel_warmup``, before the JIT monitor activates.
"""

from __future__ import annotations

from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)


def register_top_k_top_p_warmup(worker: Any) -> bool:
    """Register ``_topk_topp_kernel`` with the v2 runner's JIT warmup registry.

    Must run before ``compile_or_warm_up_model`` reaches ``kernel_warmup``.
    Returns whether the kernel was registered.
    """
    if not getattr(worker, "use_v2_model_runner", False) or not current_platform.is_cuda():
        return False
    model_runner = worker.model_runner
    if not getattr(model_runner, "is_last_pp_rank", True) or getattr(model_runner, "is_pooling_model", False):
        return False
    registry = getattr(model_runner, "jit_warmup_registry", None)
    if registry is None:
        return False

    from vllm.triton_utils import HAS_TRITON

    if not HAS_TRITON:
        return False
    from vllm.v1.sample.ops.topk_topp_triton import _topk_topp

    with registry.activate():
        _topk_topp.register_warmup()
    logger.debug("Registered the Triton top-k/top-p sampling kernel for JIT warmup.")
    return True
