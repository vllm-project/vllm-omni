# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audiogen_scripts/cfg_logits_processor.py
#   inference_scripts_vllm/audiogen_scripts/vllm_cfg_patch.py
# Divergences from the official files are commented at the divergence site.
"""Classifier-free guidance for the Audex thinker stage.

CFG pairs a conditional request with an unconditional (null-prompt)
companion in the same engine and blends their logits every step:

    blended = uncond + cfg_scale * (cond - uncond)

Both rows receive the blended logits and the sampled token is copied from
the cond row to the uncond row, so the two sequences stay token-identical.
Requests opt in via ``SamplingParams.extra_args``:

    cond:   {"cfg_scale": 1.5, "cfg_role": "cond",   "cfg_pair_id": <id>}
    uncond: {"cfg_scale": 1.5, "cfg_role": "uncond", "cfg_pair_id": <id>}

Blending is only correct when both pair members decode the same position in
the same engine step, so ``apply_cfg_patches()`` makes the vLLM scheduler
pair-aware: a lone member waits for its partner, partners stay adjacent,
their ``num_computed_tokens`` are equalized after every schedule, and they
finish together. The patch is a no-op for engines that never see a CFG
request (the pair registry stays empty).
"""

from __future__ import annotations

from collections import deque
from typing import Any

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    BatchUpdate,
    LogitsProcessor,
    MoveDirectionality,
)

logger = init_logger(__name__)

_COND = "cond"
_UNCOND = "uncond"


def _cfg_extra(request: Any, key: str) -> Any:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if extra_args:
        return extra_args.get(key)
    return None


class AudexCFGLogitsProcessor(LogitsProcessor):
    """Blend cond/uncond logits for classifier-free guidance.

    Pairs are matched by ``cfg_pair_id``. For each pair the blended logits
    are written to *both* rows so the sampler picks the same token; the
    post-sampling copy of the cond token into the uncond slot is installed
    by :meth:`_ensure_sample_patched` (which must run inside each worker
    process, hence from ``__init__`` rather than a main-process patch).
    """

    _sample_patched = False

    @classmethod
    def validate_params(cls, params: SamplingParams) -> None:
        extra_args = params.extra_args
        if not extra_args:
            return
        role = extra_args.get("cfg_role")
        if role is not None and role not in (_COND, _UNCOND):
            raise ValueError(f"cfg_role must be 'cond' or 'uncond', got '{role}'")
        scale = extra_args.get("cfg_scale")
        if scale is not None and (not isinstance(scale, int | float) or scale < 1.0):
            raise ValueError(f"cfg_scale must be >= 1.0, got {scale}")

    def __init__(self, vllm_config: VllmConfig, device: torch.device, is_pin_memory: bool) -> None:
        self._info: dict[int, dict[str, Any]] = {}
        self._output_tokens: dict[int, list[int]] = {}
        self._pairs: list[tuple[int, int, float]] = []
        self._dirty = True
        self._ensure_sample_patched()

    @classmethod
    def _ensure_sample_patched(cls) -> None:
        """Install the post-sampling cond→uncond token copy (once per process)."""
        if cls._sample_patched:
            return
        cls._sample_patched = True

        # Divergence from the official file: it patches vLLM's
        # ``GPUModelRunner._sample``, but vllm-omni's AR runner overrides
        # ``_sample``, so the base-class patch would never run for the Audex
        # thinker stage. Patch the omni AR runner instead.
        from vllm_omni.worker.gpu_ar_model_runner import GPUARModelRunner

        orig_sample = GPUARModelRunner._sample

        def _sample_with_cfg_sync(self, logits, spec_decode_metadata):
            sampler_output = orig_sample(self, logits, spec_decode_metadata)
            for proc in self.input_batch.logitsprocs.all:
                if isinstance(proc, AudexCFGLogitsProcessor) and proc._pairs:
                    sampled = sampler_output.sampled_token_ids
                    num_rows = sampled.shape[0] if hasattr(sampled, "shape") else len(sampled)
                    for cond_idx, uncond_idx, _ in proc._pairs:
                        # Steps that sample fewer rows than the persistent
                        # batch (pure-prefill / partially scheduled steps)
                        # have nothing to sync for this pair; the scheduler
                        # pair patches keep both members step-locked on the
                        # decode steps that matter.
                        if cond_idx < num_rows and uncond_idx < num_rows:
                            sampled[uncond_idx] = sampled[cond_idx]
                    break
            return sampler_output

        _sample_with_cfg_sync._audex_cfg_sync = True  # type: ignore[attr-defined]
        GPUARModelRunner._sample = _sample_with_cfg_sync
        logger.info("AudexCFGLogitsProcessor: patched GPUARModelRunner._sample for CFG token sync")

    def is_argmax_invariant(self) -> bool:
        return False

    def _reset(self) -> None:
        self._info.clear()
        self._output_tokens.clear()
        self._pairs.clear()
        self._dirty = True

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        if batch_update is None:
            return

        for idx in batch_update.removed:
            self._info.pop(idx, None)
            self._output_tokens.pop(idx, None)

        if not self._info and batch_update.added:
            self._reset()

        for idx, params, _, output_token_ids in batch_update.added:
            extra_args = params.extra_args if params else None
            if extra_args and extra_args.get("cfg_role") in (_COND, _UNCOND):
                self._info[idx] = {
                    "role": extra_args["cfg_role"],
                    "cfg_scale": float(extra_args.get("cfg_scale", 1.0)),
                    "pair_id": extra_args.get("cfg_pair_id"),
                }
                self._output_tokens[idx] = output_token_ids
            else:
                self._info.pop(idx, None)
                self._output_tokens.pop(idx, None)
        self._dirty = True

        if self._info:
            for src, dst, direction in batch_update.moved:
                src_info = self._info.pop(src, None)
                dst_info = self._info.pop(dst, None)
                src_tokens = self._output_tokens.pop(src, None)
                dst_tokens = self._output_tokens.pop(dst, None)
                if src_info is not None:
                    self._info[dst] = src_info
                if src_tokens is not None:
                    self._output_tokens[dst] = src_tokens
                if direction == MoveDirectionality.SWAP:
                    if dst_info is not None:
                        self._info[src] = dst_info
                    if dst_tokens is not None:
                        self._output_tokens[src] = dst_tokens
            self._dirty = True

    def _rebuild_pairs(self) -> None:
        by_pair: dict[str, dict[str, tuple[int, float]]] = {}
        for idx, info in self._info.items():
            pair_id = info.get("pair_id")
            if pair_id is None:
                continue
            by_pair.setdefault(pair_id, {})[info["role"]] = (idx, info["cfg_scale"])

        self._pairs = [
            (roles[_COND][0], roles[_UNCOND][0], roles[_COND][1])
            for roles in by_pair.values()
            if _COND in roles and _UNCOND in roles
        ]
        self._dirty = False

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self._info:
            return logits

        if self._dirty:
            self._rebuild_pairs()

        num_rows = logits.shape[0]
        for cond_idx, uncond_idx, cfg_scale in self._pairs:
            # Rows beyond this step's logits (pure-prefill / partially
            # scheduled steps) have nothing to blend; decode steps carry
            # both pair rows thanks to the scheduler pair patches.
            if cond_idx >= num_rows or uncond_idx >= num_rows:
                continue
            blended = logits[uncond_idx] + cfg_scale * (logits[cond_idx] - logits[uncond_idx])
            logits[cond_idx] = blended
            logits[uncond_idx] = blended

        return logits


# ---------------------------------------------------------------------------
# Scheduler pair synchronization
# ---------------------------------------------------------------------------


def _wait_queues(scheduler: Any) -> list[Any]:
    # Divergence from the official (vLLM 0.20-era) file: vLLM 0.24 parks
    # structurally not-ready requests in a second ``skipped_waiting`` queue;
    # pair-hold must cover both queues or a lone member parked there would
    # be admitted without its partner.
    queues = [scheduler.waiting]
    skipped = getattr(scheduler, "skipped_waiting", None)
    if skipped is not None:
        queues.append(skipped)
    return queues


def _pair_complete(scheduler: Any, request_id: str, blocked_ids: set[str] | None = None) -> bool:
    pair_id = scheduler._cfg_req_to_pair.get(request_id)
    if pair_id is None:
        return True
    roles = scheduler._cfg_pairs.get(pair_id, {})
    if len(roles) != 2 or not all(rid in scheduler.requests for rid in roles.values()):
        return False
    if blocked_ids:
        # A partner parked in skipped_waiting (e.g. waiting for remote KVs)
        # may fail promotion this step while this member gets scheduled from
        # waiting — hold this member until the partner is schedulable.
        for rid in roles.values():
            if rid != request_id and rid in blocked_ids:
                return False
    return True


# Schedule steps a lone pair member may wait for its partner. Partners arrive
# within a handful of steps in practice; exhausting this budget means the
# companion was never created (e.g. prompt expansion failed), and the request
# is released to decode unguided instead of hanging forever.
_MAX_PAIR_HOLD_STEPS = 512


def _drop_stale_pair(scheduler: Any, request_id: str) -> None:
    pair_id = scheduler._cfg_req_to_pair.get(request_id)
    if pair_id is None:
        return
    for rid in scheduler._cfg_pairs.pop(pair_id, {}).values():
        scheduler._cfg_req_to_pair.pop(rid, None)
    logger.warning(
        "CFG pair %s: partner of %s never arrived after %d schedule steps; releasing the request unguided",
        pair_id,
        request_id,
        _MAX_PAIR_HOLD_STEPS,
    )


def _hold_incomplete_pairs(scheduler: Any) -> list[tuple[Any, Any]]:
    """Pull CFG requests whose partner is not yet admitted out of the wait queues.

    The engine schedules as soon as the first member of a pair arrives over
    IPC; without the hold, the lone member prefills one step early and the
    pair is permanently offset by one token. Held requests are re-prepended
    after the schedule step.
    """
    hold_counts: dict[str, int] = getattr(scheduler, "_cfg_hold_counts", None) or {}
    scheduler._cfg_hold_counts = hold_counts

    skipped = getattr(scheduler, "skipped_waiting", None)
    blocked_ids = {req.request_id for req in list(skipped)} if skipped is not None else set()

    held: list[tuple[Any, Any]] = []
    for queue in _wait_queues(scheduler):
        # A member is only "blocked" for its partner's sake when it sits in
        # the OTHER queue; members of the queue being scanned move together.
        partner_blockers = blocked_ids if queue is not skipped else set()
        held_here = []
        for request in list(queue):
            if _pair_complete(scheduler, request.request_id, partner_blockers):
                hold_counts.pop(request.request_id, None)
                continue
            count = hold_counts.get(request.request_id, 0) + 1
            if count > _MAX_PAIR_HOLD_STEPS:
                hold_counts.pop(request.request_id, None)
                _drop_stale_pair(scheduler, request.request_id)
                continue
            hold_counts[request.request_id] = count
            held_here.append(request)
        if held_here:
            queue.remove_requests(held_here)
            held.extend((queue, req) for req in held_here)
    return held


def _release_held(held: list[tuple[Any, Any]]) -> None:
    for queue, request in reversed(held):
        queue.prepend_request(request)


def _reorder_waiting_for_cfg(scheduler: Any) -> None:
    """Move CFG pair partners adjacent in the FCFS waiting queue."""
    waiting = scheduler.waiting
    # Only the FCFS queue (a deque) has caller-controlled ordering; priority
    # queues order themselves and pair sync then relies on hold + equalize.
    if not isinstance(waiting, deque) or len(waiting) < 2:
        return

    requests = list(waiting)
    waiting.clear()

    seen: set[str] = set()
    result: list[Any] = []
    for request in requests:
        rid = request.request_id
        if rid in seen:
            continue
        seen.add(rid)
        result.append(request)

        pair_id = scheduler._cfg_req_to_pair.get(rid)
        if pair_id is None:
            continue
        for partner_id in scheduler._cfg_pairs.get(pair_id, {}).values():
            if partner_id != rid and partner_id not in seen:
                for candidate in requests:
                    if candidate.request_id == partner_id:
                        seen.add(partner_id)
                        result.append(candidate)
                        break

    waiting.extend(result)


def _equalize_cfg_pair_progress(scheduler: Any, scheduler_output: Any) -> None:
    """Bring both members of every CFG pair to the same ``num_computed_tokens``.

    Prefix-cache hits or unequal chunked-prefill budget can leave one member
    ahead after a schedule step; reduce the faster member's allocation so
    both land on the same position. Over-allocated KV blocks remain and are
    consumed next step — nothing leaks.
    """
    for roles in scheduler._cfg_pairs.values():
        cond_id = roles.get(_COND)
        uncond_id = roles.get(_UNCOND)
        if not cond_id or not uncond_id:
            continue

        cond_sched = scheduler_output.num_scheduled_tokens.get(cond_id, 0)
        uncond_sched = scheduler_output.num_scheduled_tokens.get(uncond_id, 0)
        if cond_sched == 0 or uncond_sched == 0:
            continue

        cond_req = scheduler.requests.get(cond_id)
        uncond_req = scheduler.requests.get(uncond_id)
        if not cond_req or not uncond_req:
            continue

        if cond_req.num_computed_tokens == uncond_req.num_computed_tokens:
            continue

        target = min(cond_req.num_computed_tokens, uncond_req.num_computed_tokens)

        feasible = all(
            req.num_computed_tokens - target < sched
            for req, sched in ((cond_req, cond_sched), (uncond_req, uncond_sched))
        )
        if not feasible:
            continue

        for req_id, req, orig_sched in (
            (cond_id, cond_req, cond_sched),
            (uncond_id, uncond_req, uncond_sched),
        ):
            diff = req.num_computed_tokens - target
            if diff > 0:
                req.num_computed_tokens = target
                scheduler_output.num_scheduled_tokens[req_id] = orig_sched - diff
                scheduler_output.total_num_scheduled_tokens -= diff
                logger.debug(
                    "CFG equalize %s: %d -> %d scheduled, computed -> %d",
                    req_id,
                    orig_sched,
                    orig_sched - diff,
                    target,
                )


def _drop_split_pairs(scheduler: Any) -> None:
    """Deregister a desynced CFG pair (one member preempted / left behind).

    Manual re-preemption surgery is unsafe against the upstream scheduler.
    Instead, a pair that split anyway (KV-pressure preemption of exactly one
    member, or unrecoverable progress skew) is dropped from the registry:
    the cond request continues UNGUIDED (its logits are never blended again)
    rather than being re-paired against a desynced partner, and the
    companion's discarded output no longer matters.
    """
    if not scheduler._cfg_pairs:
        return

    running_ids = {req.request_id for req in scheduler.running}
    to_drop: list[tuple[str, str, str, str]] = []
    for pair_id, roles in scheduler._cfg_pairs.items():
        cond_id = roles.get(_COND)
        uncond_id = roles.get(_UNCOND)
        if cond_id is None or uncond_id is None:
            continue
        if cond_id not in scheduler.requests or uncond_id not in scheduler.requests:
            continue

        cond_running = cond_id in running_ids
        uncond_running = uncond_id in running_ids
        if cond_running != uncond_running:
            # Split only counts once decoding started: a lone member that has
            # computed tokens while its partner was pushed back is desynced.
            runner = scheduler.requests.get(cond_id if cond_running else uncond_id)
            if runner is not None and runner.num_computed_tokens > 0:
                to_drop.append((pair_id, cond_id, uncond_id, "one member preempted"))
            continue

        if cond_running and uncond_running:
            cond_req = scheduler.requests.get(cond_id)
            uncond_req = scheduler.requests.get(uncond_id)
            if cond_req and uncond_req and cond_req.num_computed_tokens != uncond_req.num_computed_tokens:
                logger.warning(
                    "CFG pair progress mismatch (equalize will retry): %s@%d vs %s@%d",
                    cond_id,
                    cond_req.num_computed_tokens,
                    uncond_id,
                    uncond_req.num_computed_tokens,
                )

    for pair_id, cond_id, uncond_id, reason in to_drop:
        scheduler._cfg_pairs.pop(pair_id, None)
        scheduler._cfg_req_to_pair.pop(cond_id, None)
        scheduler._cfg_req_to_pair.pop(uncond_id, None)
        logger.warning(
            "CFG pair %s split (%s); releasing cond=%s unguided, uncond=%s output is discarded",
            pair_id,
            reason,
            cond_id,
            uncond_id,
        )


_patches_applied = False


def cfg_patches_applied() -> bool:
    """True once :func:`apply_cfg_patches` has patched the scheduler in this process."""
    return _patches_applied


def _patch_scheduler() -> None:
    from vllm.v1.core.sched.scheduler import Scheduler

    orig_init = Scheduler.__init__
    orig_add = Scheduler.add_request
    orig_finish = Scheduler.finish_requests
    orig_schedule = Scheduler.schedule

    def _init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._cfg_pairs: dict[str, dict[str, str]] = {}
        self._cfg_req_to_pair: dict[str, str] = {}

    def _add(self, request: Any) -> None:
        orig_add(self, request)
        pair_id = _cfg_extra(request, "cfg_pair_id")
        role = _cfg_extra(request, "cfg_role")
        if pair_id and role:
            self._cfg_pairs.setdefault(pair_id, {})[role] = request.request_id
            self._cfg_req_to_pair[request.request_id] = pair_id
            # An asymmetric prefix-cache hit (the cond prompt shares cached
            # branches; the <unk> null prompt does not) can advance one
            # member past what post-schedule equalization can claw back.
            # Skip prefix-cache reads for CFG requests entirely.
            if hasattr(request, "skip_reading_prefix_cache"):
                request.skip_reading_prefix_cache = True

    def _finish(self, request_ids: Any, finished_status: Any) -> Any:
        if request_ids is None:
            result = orig_finish(self, request_ids, finished_status)
            self._cfg_pairs.clear()
            self._cfg_req_to_pair.clear()
            return result

        if isinstance(request_ids, str):
            request_ids = {request_ids}
        else:
            request_ids = set(request_ids)

        # Finish both members together: a surviving partner would decode
        # unpaired garbage (its logits are never blended again).
        partner_ids: set[str] = set()
        for req_id in request_ids:
            pair_id = self._cfg_req_to_pair.get(req_id)
            if pair_id is None:
                continue
            for rid in self._cfg_pairs.get(pair_id, {}).values():
                if rid != req_id and rid in self.requests:
                    partner_ids.add(rid)

        all_ids = request_ids | partner_ids
        result = orig_finish(self, all_ids, finished_status)

        for req_id in all_ids:
            pair_id = self._cfg_req_to_pair.pop(req_id, None)
            if pair_id:
                self._cfg_pairs.pop(pair_id, None)
        return result

    def _schedule(self, *args: Any, **kwargs: Any) -> Any:
        if not self._cfg_pairs:
            return orig_schedule(self, *args, **kwargs)

        _reorder_waiting_for_cfg(self)
        held = _hold_incomplete_pairs(self)

        # Chunked prefill can split one pair member's prompt across steps
        # while the other fits in a single step; halving the long-prefill
        # threshold makes both chunk, and equalize aligns the remainder.
        scheduler_config = getattr(self, "scheduler_config", None)
        orig_threshold = getattr(scheduler_config, "long_prefill_token_threshold", None)
        if scheduler_config is not None and orig_threshold is not None:
            scheduler_config.long_prefill_token_threshold = self.max_num_scheduled_tokens // 2
        try:
            scheduler_output = orig_schedule(self, *args, **kwargs)
        finally:
            if scheduler_config is not None and orig_threshold is not None:
                scheduler_config.long_prefill_token_threshold = orig_threshold
            _release_held(held)

        _equalize_cfg_pair_progress(self, scheduler_output)
        _drop_split_pairs(self)
        return scheduler_output

    _schedule._audex_cfg_patched = True  # type: ignore[attr-defined]

    Scheduler.__init__ = _init
    Scheduler.add_request = _add
    Scheduler.finish_requests = _finish
    Scheduler.schedule = _schedule


def apply_cfg_patches() -> None:
    """Make the vLLM v1 scheduler CFG-pair-aware. Idempotent, per process.

    Must run in the engine-core process before ``Scheduler`` is constructed
    (the ``__init__`` wrapper installs the pair registry).
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    logger.info("Applying Audex CFG scheduler patches to vLLM v1")
    _patch_scheduler()
