# SPDX-License-Identifier: Apache-2.0
"""Multi-step decode window planning for AR stages on async scheduling.

Small autoregressive stages (e.g. the MiniCPM-o 4.5 Talker) spend more host
time per decode step (scheduling, input preparation, attention metadata,
sampling bookkeeping, inter-stage IPC) than device time on the forward
itself.  This module lets one engine step cover ``K`` decode steps for the
whole batch, amortizing the per-step host cost over ``K`` tokens:

1. ``num_scheduled_tokens`` for the batch is bumped from 1 to K after the
   base scheduler already ran (a post-schedule patch).  The extra K-1 KV
   slots are allocated here (``allocate_slots``) and the new block ids are
   appended to ``scheduled_cached_reqs.new_block_ids`` so the worker's
   block table covers every slot the window will write.
2. ``request.num_output_placeholders`` and ``request.num_computed_tokens``
   are inflated by K-1.  The async-scheduling accounting then treats the
   window exactly like a step that samples K tokens per request: the K
   reported tokens drive ``num_output_placeholders`` back to zero, while
   ``update_from_output`` reconciles the shortfall when the runner produced
   fewer tokens than planned (early stop or single-step fallback), so the
   engine's confirmed ``num_computed_tokens`` always matches the KV slots
   actually written.
3. The inflated placeholder count doubles as the fence: while a window is
   in flight, ``num_tokens_with_spec + placeholders - computed == 0`` keeps
   the request out of subsequent schedules until its K tokens are
   reconciled.

Concurrency safety: the window freezes the batch composition for K steps,
so a request waiting in the scheduler's waiting queue would be blocked from
admission (its prefill cannot be scheduled mid-window).  The planner
therefore refuses to open a window while the waiting queue is non-empty --
new work is admitted first, and windows resume once the queue drains.  This
keeps multi-concurrency latency identical to the baseline while still
amortizing host cost over the (common) steady state where every admitted
request is decoding.

Models opt in by declaring ``supports_multi_step_decode = True`` on the
model class.  The contract behind the flag:

* ``model.preprocess(input_ids, input_embeds, **info)`` builds the decode
  embedding of every request purely from the request-local state chain
  (the previously sampled token and per-request caches) and advances that
  state in place through ``make_omni_output``;
* intermediate steps need no per-step OmniOutput wire wrapping -- hidden
  states and codec deltas can be accumulated and shipped once at window
  end.

Every refusal path is fail-closed: the patch is skipped entirely and the
normal single-step engine loop runs.  Rollback: leave
``multi_step_decode_steps`` at 0 (the default) in the deploy config.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)

# Debug/override knob: when set, takes precedence over the deploy-config
# value (clamped the same way).  Useful for quick A/B experiments without
# editing the deploy yaml.
ENV_MULTI_STEP_STEPS = "VLLM_OMNI_MULTI_STEP_STEPS"

# Upper bound on the window size.  The amortization benefit saturates well
# before this (host cost becomes negligible vs. accumulated device time)
# while the admission-blocking tail grows linearly with K.
MAX_MULTI_STEP_STEPS = 16

# Cache of architecture -> capability lookups so the registry is only
# touched once per architecture (the registry import pulls in the whole
# model registry module graph).
_capability_cache: dict[str, bool] = {}


def resolve_multi_step_steps(model_config: Any) -> int:
    """Resolve the window size for a stage from config (+env override).

    Returns 0 when the window is disabled and otherwise K clamped to
    ``[1, MAX_MULTI_STEP_STEPS]``.
    """
    raw = os.environ.get(ENV_MULTI_STEP_STEPS)
    if raw is not None:
        try:
            value = int(raw, 10)
        except ValueError:
            logger.warning_once(
                "Invalid %s=%r; ignoring it.", ENV_MULTI_STEP_STEPS, raw
            )
        else:
            if value <= 0:
                return 0
            return min(value, MAX_MULTI_STEP_STEPS)
    value = int(getattr(model_config, "multi_step_decode_steps", 0) or 0)
    if value <= 0:
        return 0
    return min(value, MAX_MULTI_STEP_STEPS)


def model_supports_multi_step(model_config: Any) -> bool:
    """True when the stage's model class declares the multi-step contract.

    Resolves the architecture through the omni model registry and reads the
    ``supports_multi_step_decode`` class attribute; unknown or transformers
    fallback architectures simply do not declare it.

    Stage-dispatching wrapper architectures (one class backing both the
    thinker and talker stages) declare the capability per stage via
    ``supports_multi_step_stages`` — a tuple of ``model_stage`` values that
    host a multi-step-capable AR model.  When that attribute is present it
    takes precedence: the stage qualifies only if its ``model_stage`` is
    listed, so the thinker stage of the same wrapper never opens windows.
    """
    archs = list(getattr(model_config, "architectures", None) or ())
    stage = getattr(model_config, "model_stage", None)
    for arch in archs:
        cached = _capability_cache.get((arch, stage))
        if cached is not None:
            if cached:
                return True
            continue
        supported = False
        try:
            from vllm_omni.model_executor.models.registry import OmniModelRegistry

            model_cls, _ = OmniModelRegistry.resolve_model_cls([arch], model_config)
            capable_stages = getattr(model_cls, "supports_multi_step_stages", None)
            if capable_stages is not None:
                supported = stage in capable_stages
            else:
                supported = bool(
                    getattr(model_cls, "supports_multi_step_decode", False)
                )
        except Exception:
            # Registry misses and load errors both mean "not declared".
            supported = False
        _capability_cache[(arch, stage)] = supported
        if supported:
            return True
    return False


def scheduler_allows_multi_step(scheduler: Any) -> bool:
    """Static configuration gate evaluated once per ``schedule()`` call.

    Only single-rank (TP/PP/DP/PCP/DCP == 1), spec-decode-free, LoRA-free
    AR stages on async scheduling are eligible.  Anything the runner cannot
    reproduce bit-identically (KV-transfer criteria, routed experts,
    encoder-decoder, mamba-aligned caches) is refused here.
    """
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None:
        return False
    # The window executor currently ships with the NPU AR runner only (see
    # vllm_omni/platforms/npu/worker/multi_step_decode.py).  Planning windows
    # for a runner without the execute_model hook would silently turn the
    # K-token scheduling into a plain multi-token decode and corrupt outputs,
    # so platforms without the hook are refused here (fail-closed).  Porting
    # to another platform: add the execute_model/sample_tokens hooks to its
    # runner, then relax this check.
    if getattr(current_platform, "device_type", "") != "npu":
        return False
    model_config = vllm_config.model_config
    if not model_supports_multi_step(model_config):
        return False
    if not getattr(scheduler.scheduler_config, "async_scheduling", False):
        return False
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.pipeline_parallel_size != 1
        or parallel_config.tensor_parallel_size != 1
        or parallel_config.data_parallel_size != 1
        or getattr(parallel_config, "pcp_size", 1) != 1
        or getattr(parallel_config, "dcp_size", 1) != 1
    ):
        return False
    if getattr(scheduler, "num_spec_tokens", 0) != 0:
        return False
    if getattr(scheduler, "kv_transfer_criteria", None) is not None:
        return False
    if getattr(scheduler, "lora_config", None) is not None:
        return False
    if getattr(model_config, "is_encoder_decoder", False):
        return False
    if getattr(model_config, "enable_return_routed_experts", False):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if getattr(cache_config, "mamba_cache_mode", "off") == "align":
        return False
    return True


def _request_admits_multi_step(request: Any) -> bool:
    """Per-request eligibility for one multi-step window."""
    if request is None or request.is_finished():
        return False
    if getattr(request, "status", None) != RequestStatus.RUNNING:
        return False
    # Decode phase only: the window writes K contiguous post-prompt slots and
    # cannot interleave with chunked prefill or prefix-cache (re)computation.
    if request.num_computed_tokens < request.num_prompt_tokens:
        return False
    if getattr(request, "has_encoder_inputs", False):
        return False
    if getattr(request, "use_structured_output", False):
        return False
    if getattr(request, "pooling_params", None) is not None:
        return False
    params = request.sampling_params
    if params is None:
        return False
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if getattr(params, "bad_words_token_ids", None):
        return False
    if getattr(params, "allowed_token_ids", None):
        return False
    # Value-dependent logits processors must stay off: the window keeps the
    # placeholder (-1) bookkeeping in step, but only length-based processors
    # (min_tokens) are value-independent.
    if getattr(params, "frequency_penalty", 0.0) != 0.0:
        return False
    if getattr(params, "presence_penalty", 0.0) != 0.0:
        return False
    if getattr(params, "repetition_penalty", 1.0) != 1.0:
        return False
    return True


def _request_window_budget(request: Any, max_model_len: int) -> int:
    """Largest K this request can host: min(K, max_tokens left, context)."""
    params = request.sampling_params
    max_tokens = int(getattr(params, "max_tokens", 0) or 0)
    remaining = max_tokens - len(request.output_token_ids) if max_tokens else 0
    context_room = max_model_len - request.num_tokens
    return min(remaining, context_room)


def plan_multi_step_window(scheduler: Any, scheduler_output: Any, steps: int) -> bool:
    """Rewrite ``scheduler_output`` in place into a multi-step decode window.

    Must be called right after the base ``schedule()`` produced a plain
    single-token decode step.  Returns True when the output was rewritten;
    on any refusal the output and the request accounting are left untouched
    so the caller falls back to the normal single-step path (the scheduler
    side of ``update_from_output`` reconciles the reservation shortfall if
    the worker ever declines a plan that was emitted).
    """
    if steps < 2:
        return False
    # Admission gate: a window freezes the batch composition, which would
    # delay the prefill of anything already waiting.  Admit new work first;
    # windows resume once the waiting queue drains.
    if getattr(scheduler, "waiting", None):
        return False
    num_scheduled = scheduler_output.num_scheduled_tokens
    if not num_scheduled:
        return False
    if scheduler_output.scheduled_new_reqs:
        return False
    if scheduler_output.scheduled_spec_decode_tokens:
        return False
    if scheduler_output.scheduled_encoder_inputs:
        return False
    if getattr(scheduler_output, "has_structured_output_requests", False):
        return False
    if getattr(scheduler_output, "pending_structured_output_tokens", False):
        return False

    cached_reqs = scheduler_output.scheduled_cached_reqs
    cached_index = {req_id: i for i, req_id in enumerate(cached_reqs.req_ids)}
    windowed: list[tuple[str, Any]] = []
    for req_id, num_tokens in num_scheduled.items():
        # The base scheduler must have scheduled exactly one decode token.
        if num_tokens != 1:
            return False
        request = scheduler.requests.get(req_id)
        if not _request_admits_multi_step(request):
            return False
        if req_id not in cached_index:
            return False
        windowed.append((req_id, request))

    if not windowed:
        return False

    max_model_len = scheduler.max_model_len
    window_k = steps
    for _, request in windowed:
        window_k = min(window_k, _request_window_budget(request, max_model_len))
    if window_k < 2:
        return False

    extra = window_k - 1
    extra_blocks: dict[str, Any] = {}
    for req_id, request in windowed:
        # Extend the block allocation to cover the K-1 extra KV slots the
        # window will write.  On failure allocate_slots leaves state intact
        # and returns None; the window is refused for the whole batch.
        blocks = scheduler.kv_cache_manager.allocate_slots(
            request, extra, num_lookahead_tokens=0
        )
        extra_blocks[req_id] = blocks
        if blocks is None:
            logger.debug(
                "Multi-step window refused: request %s cannot host %d extra KV slots",
                req_id,
                extra,
            )

    # Communicate whatever allocations succeeded as look-ahead block rows so
    # the scheduler/worker block tables never diverge, even when the window
    # itself is refused below.
    for i, req_id in enumerate(cached_reqs.req_ids):
        blocks = extra_blocks.get(req_id)
        if blocks is None:
            continue
        new_ids = blocks.get_block_ids()
        if not any(new_ids):
            continue
        current = cached_reqs.new_block_ids[i]
        if current is None:
            cached_reqs.new_block_ids[i] = new_ids
        else:
            cached_reqs.new_block_ids[i] = tuple(
                old + new for old, new in zip(current, new_ids)
            )

    if any(blocks is None for blocks in extra_blocks.values()):
        return False

    # Commit the window: K scheduled tokens per request and K in-flight
    # output placeholders so the engine-side accounting closes exactly when
    # the runner reports the K sampled tokens.
    for req_id, request in windowed:
        request.num_output_placeholders += extra
        request.num_computed_tokens += extra
        scheduler_output.num_scheduled_tokens[req_id] = window_k
        scheduler_output.multi_step_plan[req_id] = window_k
    scheduler_output.total_num_scheduled_tokens += extra * len(windowed)
    logger.debug(
        "Multi-step window scheduled: K=%d requests=%d", window_k, len(windowed)
    )
    return True


def reconcile_window_shortfall(
    request: Any, planned_steps: int, reported_tokens: int
) -> None:
    """Close the accounting gap when a windowed step produced < K tokens.

    ``AsyncScheduler._update_request_with_output`` already consumed
    ``reported_tokens`` placeholders and its ``cache_blocks`` call used
    ``num_computed_tokens - num_output_placeholders``, which equals the true
    confirmed length in both the window and fallback cases.  The remaining
    K - reported reservations (placeholders + optimistic computed tokens)
    belong to steps that never ran and are rolled back here.
    """
    shortfall = planned_steps - reported_tokens
    if shortfall <= 0:
        return
    # Placeholders hold exactly K - reported in steady state; clamp only to
    # stay defensive against unexpected interim adjustments, and roll back
    # computed_tokens by the same amount so the (computed - placeholders)
    # confirmed-length invariant is preserved.
    shortfall = min(shortfall, max(request.num_output_placeholders, 0))
    request.num_output_placeholders -= shortfall
    request.num_computed_tokens -= shortfall
