# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Native aligned recurrent-state migration for the overridden AR loop."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

    from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner


def prepare_mamba_prefix_cache(
    runner: GPUARModelRunner,
    scheduler_output: SchedulerOutput,
    num_reqs: int,
    deferred_state_corrections_fn: Callable[[], None] | None,
) -> Callable[[], None] | None:
    """Apply native align-mode migration before attention metadata is built.

    Matches vLLM's execution lifecycle: corrections must precede state copies,
    which can change accepted-token counts. Consuming the callback here avoids
    applying it twice at the end of the overridden execution loop.
    """
    policy = getattr(runner, "_omni_cache_policy", None)
    if not getattr(policy, "native_kv_cache_only", False) or runner.cache_config.mamba_cache_mode != "align":
        return deferred_state_corrections_fn

    from vllm.v1.worker import mamba_utils

    if deferred_state_corrections_fn:
        deferred_state_corrections_fn()
    bufs = runner._get_mamba_bufs()
    mamba_utils.preprocess_mamba(
        scheduler_output,
        runner.kv_cache_config,
        runner.cache_config,
        runner.mamba_state_idx,
        runner.input_batch,
        runner.requests,
        runner.compilation_config.static_forward_context,
        runner._get_mamba_state_copy_funcs(),
        bufs.preprocess,
        align_ctx=bufs.postprocess_align,
    )
    runner.num_accepted_tokens.np[:num_reqs] = runner.input_batch.num_accepted_tokens_cpu[:num_reqs]
    runner.num_accepted_tokens.copy_to_gpu(num_reqs)
    if bufs.postprocess_align is not None:
        mamba_utils.stage_postprocess_inputs_to_gpu(
            bufs.postprocess_align,
            scheduler_output,
            runner.input_batch.req_ids,
            num_reqs,
            runner.requests,
            runner.mamba_state_idx,
        )
    return None
