"""Steady-state decode input preparation for the MiniCPM-o Talker.

`_prepare_inputs` is written for the general case: chunked prefill, spec decode,
PCP/DCP, M-RoPE, LoRA, prompt embeds. It rebuilds every per-step tensor from
numpy each step and pushes ~15 separate H2D copies, because in general every
one of them can change.

On the scored simplex path none of them do. Stage 1 runs one request at a time
generating one codec frame per step for ~118 steps, and across those steps the
request set, the batch shape and the scheduled-token counts are all constant.
Only four things move: the token that was just sampled, the position, the
sequence length and the KV slot. Everything else -- `query_start_loc`,
`req_indices`, `query_pos`, `num_scheduled_tokens`, `num_accepted_tokens`,
`logits_indices`, `query_lens` -- is recomputed to the same value it already
held.

Measured in place on A3 / 910C with the stock deploy config, Seed-TTS zh,
32 requests at concurrency 1: `_prepare_inputs` runs 3591 times in stage 1 and
507 times in stage 0 per run, costing 0.95-1.24 ms a call and 127 ms of host
time per request. 98% of those calls are steps where nothing it rebuilds has
changed. This module reuses the previous step's tensors when the batch is
provably unchanged and applies only the deltas, which takes that 127 ms to
81 ms.

**The reuse rule is deliberately narrow.** The fast path runs only when the
*immediately preceding* step also prepared a pure single-token decode with the
identical request set. Any other step -- a prefill, a new request, a reordering
-- goes through the generic path, which rewrites the shared buffers, and only
the step after *that* becomes eligible again. So a cached tensor can never
outlive the buffer contents it was cached against.

There is no switch to turn this off, because there is no state in which it is
the only thing standing between a correct step and a wrong one: every gate below
returns a *reason* and the caller falls back to the generic path, so an
unrecognised configuration, an unfamiliar block-table layout or an unexpected
step shape all degrade to today's behaviour rather than to a guess.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from vllm.logger import init_logger

logger = init_logger(__name__)


class DecodeInputCache:
    """Per-runner state for the steady-state decode reuse.

    **Holds no torch tensors.** A tensor created inside `torch.inference_mode()`
    is an inference tensor; carrying one into a later step and indexing with it
    raises "Inference tensors cannot be saved for backward". Everything cached
    here is a numpy array, an int, or an enum, and every tensor the fast path
    hands back is built in the step that uses it.
    """

    __slots__ = (
        "signature",
        "cu_num_tokens",
        "num_tokens_np",
        "block_table_snapshot",
        "static_ok",
        "attn_state",
        "with_prefill",
        "fast_steps",
        "generic_steps",
        "blocked_reason",
    )

    def __init__(self) -> None:
        self.signature: tuple | None = None
        self.cu_num_tokens: np.ndarray | None = None
        self.num_tokens_np: np.ndarray | None = None
        self.block_table_snapshot: list[np.ndarray] | None = None
        self.static_ok: bool | None = None
        # Taken from the generic run rather than imported: the enum lives in
        # vllm-ascend, which a ranked run installs but does not let us patch.
        self.attn_state: Any = None
        self.with_prefill: bool = False
        self.fast_steps: int = 0
        self.generic_steps: int = 0
        self.blocked_reason: str | None = None

    def invalidate(self) -> None:
        self.signature = None


def _static_gate_block(runner: Any) -> str | None:
    """Configuration-level gates, evaluated once per runner.

    Every branch of `_prepare_inputs` that the fast path does not reproduce is
    excluded here rather than tested per step. Returns the name of the first
    gate that blocks, or None when the fast path is allowed -- a bare bool made
    a non-engaging fast path indistinguishable from a broken one.
    """
    try:
        from vllm.distributed.parallel_state import get_pp_group

        if get_pp_group().world_size > 1:
            return "pipeline_parallel"
    except Exception:
        return "no_pp_group"
    if getattr(runner, "uses_mrope", False):
        return "mrope"
    if getattr(runner, "uses_xdrope_dim", 0):
        return "xdrope"
    if getattr(runner, "pcp_size", 1) != 1 or getattr(runner, "dcp_size", 1) != 1:
        return "pcp_dcp"
    if getattr(runner, "use_cp", False):
        return "context_parallel"
    if getattr(runner, "speculative_config", None) is not None:
        return "speculative_config"
    if getattr(runner, "num_spec_tokens", 0):
        return "num_spec_tokens"
    if getattr(runner, "use_async_spec_decode", False):
        return "async_spec_decode"
    if not getattr(runner, "use_async_scheduling", False):
        return "not_async_scheduling"
    if getattr(runner, "enable_prompt_embeds", False):
        return "prompt_embeds"
    if getattr(runner, "_has_gdn", False):
        return "gdn"
    if getattr(runner, "lora_config", None) is not None:
        return "lora"
    # Deliberately *not* gated on `cascade_attn_enabled`. Cascade prefix
    # lengths are computed in `execute_model` after `_prepare_inputs` returns
    # and are consumed by `_build_attention_metadata`, which stays generic --
    # `_prepare_inputs` has no cascade branch at all. Gating on it here kept
    # the fast path off the Talker entirely (round 20260828T103028Z).
    if getattr(runner, "dynamic_eplb", False):
        return "dynamic_eplb"
    model_config = getattr(runner, "model_config", None)
    if getattr(model_config, "enable_encoder_decoder", False):
        return "encoder_decoder"
    if getattr(model_config, "is_encoder_decoder", False):
        return "is_encoder_decoder"
    if getattr(model_config, "enable_return_routed_experts", False):
        return "routed_experts"
    cache_config = getattr(runner, "cache_config", None)
    if getattr(cache_config, "mamba_cache_mode", None) == "align":
        return "mamba_align"
    try:
        from vllm_ascend.utils import lmhead_tp_enable

        if lmhead_tp_enable():
            return "lmhead_tp"
    except Exception:
        return "no_lmhead_tp_probe"
    return None


def _step_gate_block(runner: Any, scheduler_output: Any, num_scheduled_tokens: np.ndarray) -> str | None:
    """Per-step gates: this really is a plain one-token-per-request decode."""
    if scheduler_output.scheduled_spec_decode_tokens:
        return "spec_decode_tokens"
    if getattr(runner, "calculate_kv_scales", False):
        return "kv_scales"
    if getattr(runner, "num_accepted_tokens_event", None) is not None:
        return "accepted_tokens_event"
    if getattr(runner, "routed_experts_initialized", False):
        return "routed_experts"
    input_batch = runner.input_batch
    if getattr(input_batch, "req_prompt_embeds", None):
        return "req_prompt_embeds"
    if input_batch.prev_sampled_token_ids is None:
        # Nothing to scatter from; the generic path's CPU copy is required.
        return "no_prev_sampled_token_ids"
    if not input_batch.prev_req_id_to_index:
        return "no_prev_req_index"
    num_reqs = input_batch.num_reqs
    if num_reqs <= 0:
        return "no_requests"
    if int(scheduler_output.total_num_scheduled_tokens) != num_reqs:
        return "not_one_token_per_request"
    if not bool(np.all(num_scheduled_tokens[:num_reqs] == 1)):
        return "not_one_token_per_request"
    # A step where any request is still consuming its prompt is not a decode.
    if bool(np.any(input_batch.num_computed_tokens_cpu[:num_reqs] == 0)):
        return "prefill_in_batch"
    return None


def _log_block_once(cache: DecodeInputCache, reason: str) -> None:
    """Say once why the fast path never engaged; silence is the worse failure."""
    if cache.blocked_reason == reason:
        return
    cache.blocked_reason = reason
    logger.info("[minicpmo] fast decode prep not engaged: %s", reason)


def _block_tables_unchanged(runner: Any, cache: DecodeInputCache, num_reqs: int) -> bool:
    """True when every block-table row is byte-identical to the last committed one.

    A decode allocates a new block once per `block_size` tokens, so for 127 of
    every 128 Talker steps the table the generic path re-uploads is the table
    already on the device. Returns False -- and refreshes the snapshot -- if it
    cannot see the rows, so an unrecognised block-table layout just keeps
    committing every step.
    """
    tables = getattr(runner.input_batch.block_table, "block_tables", None)
    if not tables:
        return False
    views = []
    for table in tables:
        rows = getattr(getattr(table, "block_table", None), "np", None)
        if rows is None:
            return False
        views.append(rows[:num_reqs])
    previous = cache.block_table_snapshot
    if previous is not None and len(previous) == len(views):
        if all(np.array_equal(p, v) for p, v in zip(previous, views)):
            return True
    cache.block_table_snapshot = [v.copy() for v in views]
    return False


def snapshot_block_tables(runner: Any, cache: DecodeInputCache, num_reqs: int) -> None:
    """Record the rows the generic path just committed, so the next fast step
    can tell whether anything moved."""
    tables = getattr(runner.input_batch.block_table, "block_tables", None)
    if not tables:
        cache.block_table_snapshot = None
        return
    views = []
    for table in tables:
        rows = getattr(getattr(table, "block_table", None), "np", None)
        if rows is None:
            cache.block_table_snapshot = None
            return
        views.append(rows[:num_reqs].copy())
    cache.block_table_snapshot = views


def signature_of(runner: Any) -> tuple:
    input_batch = runner.input_batch
    return (input_batch.num_reqs, tuple(input_batch.req_ids))


def try_fast_prepare(
    runner: Any,
    scheduler_output: Any,
    num_scheduled_tokens: np.ndarray,
) -> tuple | None:
    """Prepare a steady-state decode step, or return None to use the generic path."""
    cache: DecodeInputCache = runner._decode_input_cache
    if cache.static_ok is None:
        block = _static_gate_block(runner)
        cache.static_ok = block is None
        if block is not None:
            logger.info("[minicpmo] fast decode prep disabled for this stage: %s", block)
    if not cache.static_ok:
        return None
    block = _step_gate_block(runner, scheduler_output, num_scheduled_tokens)
    if block is not None:
        _log_block_once(cache, block)
        cache.invalidate()
        return None
    if cache.signature is None or cache.signature != signature_of(runner):
        # The previous step did not leave the shared buffers in a shape this
        # step can inherit. Fall through; `note_generic` re-arms the cache.
        return None

    input_batch = runner.input_batch
    num_reqs = input_batch.num_reqs
    total = num_reqs

    # The block table grows once per `block_size` tokens under a decode, so
    # this is a delta -- but almost always a no-op delta, and the upload is
    # ~0.15 ms of the step.
    if not _block_tables_unchanged(runner, cache, num_reqs):
        input_batch.block_table.commit_block_table(num_reqs)

    runner.with_prefill = cache.with_prefill
    runner.attn_state = cache.attn_state

    computed_cpu_tensor = input_batch.num_computed_tokens_cpu_tensor
    # numpy, not torch. Nothing here may outlive the step that made it: a torch
    # tensor born inside `torch.inference_mode()` is an *inference tensor*, and
    # a later step indexing with it raises "Inference tensors cannot be saved
    # for backward". Only plain arrays and scalars are safe to carry across
    # steps -- round 20260828T100420Z died on exactly this.
    np.add(
        input_batch.num_computed_tokens_cpu[:num_reqs],
        np.int32(1),
        out=runner.optimistic_seq_lens_cpu.numpy()[:num_reqs],
    )

    # `prev_positions` feeds the scatter in `_prepare_input_ids`; both are
    # cheap and both depend on this step's request order.
    runner._compute_prev_positions(num_reqs)
    runner._prepare_input_ids(scheduler_output, num_reqs, total, cache.cu_num_tokens)

    runner.num_computed_tokens[:num_reqs].copy_(computed_cpu_tensor[:num_reqs], non_blocking=True)
    # One scheduled token per request means `query_pos` is all zeros and
    # `req_indices` is arange(num_reqs), so both gathers collapse to a slice.
    runner.positions[:total] = runner.num_computed_tokens[:num_reqs].to(torch.int64)
    runner.seq_lens[:num_reqs] = runner.num_computed_tokens[:num_reqs] + 1

    input_batch.block_table.compute_slot_mapping(
        num_reqs,
        runner.query_start_loc.gpu[: num_reqs + 1],
        runner.positions[:total],
    )

    # A request whose sampled token would land past its own length must be
    # discarded. In steady state nothing is discarded, but the compare is a
    # handful of CPU elements and the cost of being wrong is a wrong token.
    num_tokens_np = cache.num_tokens_np
    for i, req_id in enumerate(input_batch.req_ids[:num_reqs]):
        num_tokens_np[i] = runner.requests[req_id].num_tokens
    discard_mask = runner.optimistic_seq_lens_cpu[:num_reqs].numpy() < num_tokens_np[:num_reqs]
    if bool(discard_mask.any()):
        cache.invalidate()
        return None
    if runner.num_discarded_requests:
        runner.num_discarded_requests = 0
        runner.discard_request_mask.np[:num_reqs] = False
        runner.discard_request_mask.copy_to_gpu(num_reqs)

    # Both are rebuilt, not cached, for the reason in the note above. One
    # scheduled token per request makes `logits_indices` the last (only)
    # position of each request, which is what the generic path computes too.
    logits_indices = runner.query_start_loc.gpu[1 : num_reqs + 1] - 1
    runner.query_lens = torch.from_numpy(num_scheduled_tokens[:num_reqs])
    runner.logits_indices = logits_indices
    cache.fast_steps += 1
    if cache.fast_steps == 1:
        logger.info("[minicpmo] fast decode prep engaged (num_reqs=%d)", num_reqs)
    return logits_indices, None, total


def note_generic(
    runner: Any,
    scheduler_output: Any,
    num_scheduled_tokens: np.ndarray,
    result: tuple,
) -> None:
    """Arm the cache from a generic run, when that run was itself a plain decode."""
    cache: DecodeInputCache = runner._decode_input_cache
    cache.generic_steps += 1
    if not cache.static_ok:
        return
    _logits_indices, spec_decode_metadata, total = result
    if spec_decode_metadata is not None:
        _log_block_once(cache, "spec_decode_metadata")
        cache.invalidate()
        return
    block = _step_gate_block(runner, scheduler_output, num_scheduled_tokens)
    if block is not None:
        _log_block_once(cache, block)
        cache.invalidate()
        return
    num_reqs = runner.input_batch.num_reqs
    if int(total) != num_reqs:
        _log_block_once(cache, "total_tokens_ne_num_reqs")
        cache.invalidate()
        return
    if runner.num_discarded_requests:
        # Something in this batch is not sampled; the fast path assumes nothing is.
        _log_block_once(cache, "discarded_requests")
        cache.invalidate()
        return
    if runner.with_prefill:
        _log_block_once(cache, "with_prefill")
        cache.invalidate()
        return
    cache.attn_state = runner.attn_state
    cache.with_prefill = runner.with_prefill
    cache.cu_num_tokens = np.arange(1, num_reqs + 1, dtype=np.int32)
    cache.num_tokens_np = np.zeros(num_reqs, dtype=np.int32)
    snapshot_block_tables(runner, cache, num_reqs)
    cache.signature = signature_of(runner)
