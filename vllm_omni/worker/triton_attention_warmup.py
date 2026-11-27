# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compile the Triton unified-attention prefill kernel at startup.

With the ``TRITON_ATTN`` backend (the Qwen3-Omni default on ROCm),
``kernel_unified_attention`` takes its 2D path for steps whose longest query
exceeds 8 tokens. vLLM's dummy runs spread their tokens over many one- or
two-token requests, so they only launch the 3D (decode) path, and the first
real prefill JIT-compiles the 2D kernel mid-request::

    Triton kernel JIT compilation during inference: kernel_unified_attention

Triton also specializes ``num_seqs == 1``, so a lone prefill (a realtime
session's first turn) and a prefill batched with other requests compile
separately. This module runs both before vLLM activates its JIT monitor,
reusing vLLM's mixed prefill+decode warmup, which vLLM itself only runs for
FlashInfer.
"""

from __future__ import annotations

from typing import Any

import torch
from vllm.logger import init_logger

logger = init_logger(__name__)

# Longer than the 8-token query limit of the 3D path; the 2D kernel's tile
# sizes do not depend on the prefill length.
_WARMUP_TOKENS = 16
# The "_warmup_" prefix makes omni runners skip stage-input preprocessing
# (vllm_omni.worker_v2.warmup.is_warmup_request).
LONE_PREFILL_REQ_ID = "_warmup_triton_attention_prefill_"


def _is_triton_attention(backend: Any) -> bool:
    try:
        return backend.get_name() == "TRITON_ATTN"
    except NotImplementedError:
        return False


def warm_up_triton_attention_prefill(worker: Any) -> bool:
    """Run lone and mixed prefill steps if any attention group uses ``TRITON_ATTN``.

    Must run before vLLM activates the JIT monitor at the end of
    ``compile_or_warm_up_model``. Returns whether the steps ran.
    """
    from vllm.distributed.parallel_state import get_pp_group
    from vllm.model_executor.warmup.attention_warmup import mixed_batch_attention_warmup

    runner = worker.model_runner
    eligible = int(
        not runner.is_pooling_model
        and any(_is_triton_attention(group.backend) for groups in runner.attn_groups for group in groups)
    )
    # The step goes through worker.execute_model, which joins pipeline-parallel
    # send/recv, so every PP rank must make the same decision.
    pp_group = get_pp_group()
    if pp_group.world_size > 1:
        eligibility = torch.tensor([eligible], dtype=torch.int32, device="cpu")
        pp_group.cpu_all_reduce_(eligibility)
        eligible = int(eligibility.item())
    if not eligible:
        return False
    logger.info("Warming up Triton attention prefill.")
    _run_lone_prefill(worker, _WARMUP_TOKENS)
    # The same prefill next to a decode request (num_seqs > 1).
    mixed_batch_attention_warmup(worker, num_tokens=_WARMUP_TOKENS)
    return True


def _run_lone_prefill(worker: Any, num_tokens: int) -> None:
    """Run one step with a single ``num_tokens`` prefill request.

    Mirrors vLLM's ``run_context_prefill_warmup`` without cached context:
    vLLM's own warmups always pair a prefill with a decode request, or keep
    each query at a couple of tokens.
    """
    from vllm import SamplingParams
    from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
    from vllm.v1.worker.extensible_kv_cache import ensure_kv_cache_blocks, num_committable_kv_blocks
    from vllm.v1.worker.gpu.warmup import _run_step, _warmup_block_counter, _warmup_token_ids

    runner = worker.model_runner
    num_tokens = min(num_tokens, runner.max_model_len - 1, runner.max_num_tokens)
    block_count = _warmup_block_counter(runner)
    block_counts = [block_count(num_tokens, g.kv_cache_spec) for g in runner.kv_cache_config.kv_cache_groups]
    if num_committable_kv_blocks(runner) <= sum(block_counts):
        logger.warning("Skipping the lone Triton attention prefill warmup: too few KV blocks.")
        return
    ensure_kv_cache_blocks(runner, 1 + sum(block_counts))

    next_block_id = 1
    block_ids = []
    for n in block_counts:
        block_ids.append(list(range(next_block_id, next_block_id + n)))
        next_block_id += n

    req_id = LONE_PREFILL_REQ_ID
    token_ids = _warmup_token_ids(runner, num_tokens)
    prefill_output = SchedulerOutput.make_empty()
    prefill_output.num_spec_tokens_to_schedule = runner.num_speculative_steps
    prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=req_id,
            prompt_token_ids=token_ids,
            mm_features=[],
            sampling_params=SamplingParams(max_tokens=1, temperature=0.0),
            pooling_params=None,
            block_ids=tuple(block_ids),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=token_ids,
        )
    ]
    prefill_output.num_scheduled_tokens = {req_id: num_tokens}
    prefill_output.total_num_scheduled_tokens = num_tokens
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {req_id}

    runner.kv_connector.set_disabled(True)
    try:
        _run_step(worker.execute_model, worker.sample_tokens, prefill_output)
        _run_step(worker.execute_model, None, cleanup_output)
    finally:
        runner.kv_connector.set_disabled(False)
