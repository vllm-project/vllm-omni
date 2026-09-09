# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Adapted from nvidia/Nemotron-Labs-Audex-2B (Apache-2.0):
#   inference_scripts_vllm/audiogen_scripts/vllm_cfg_patch.py
# Divergences from the official file are commented at the divergence site.
"""Model-neutral classifier-free-guidance request pairing for the vLLM v1 scheduler.

A guided request decodes together with an unconditional companion so their
logits can be blended every step. Blending is only correct when both members
decode the same position in the same engine step, so :func:`apply_cfg_patches`
makes the scheduler pair-aware: a lone member waits for its partner, partners
stay adjacent in the waiting queue, their ``num_computed_tokens`` are equalized
after every schedule, and they finish together.

Requests opt in through ``SamplingParams.extra_args``::

    cond:   {"cfg_role": "cond",   "cfg_pair_id": <id>}
    uncond: {"cfg_role": "uncond", "cfg_pair_id": <id>}

``cfg_pair_id`` is optional. A model whose serving layer cannot mint a pair id
(the companion request id is only known inside the engine) may send ``cfg_role``
alone; the pair id is then derived from the request id, stripping the
companion's registered suffix (see :func:`register_cfg_uncond_suffix`).

Logit blending itself is the caller's concern: this module only keeps the two
rows step-locked. The patch is a no-op for engines that never see a CFG request
(the pair registry stays empty).
"""

from __future__ import annotations

from collections import deque
from typing import Any

from vllm.logger import init_logger

logger = init_logger(__name__)

__all__ = [
    "apply_cfg_patches",
    "cfg_patches_applied",
    "cfg_uncond_suffixes",
    "register_cfg_uncond_suffix",
]

_COND = "cond"
_UNCOND = "uncond"

# Request-id suffix appended to the companion request by the Audex stage input
# processor (``CFG_UNCOND_SUFFIX`` in
# vllm_omni/model_executor/stage_input_processors/audex.py). Models that mint
# the companion request id differently register their own suffix.
_UNCOND_REQUEST_ID_SUFFIXES: set[str] = {"__cfg_uncond"}


def register_cfg_uncond_suffix(suffix: str) -> None:
    """Register a companion request-id suffix used to derive a missing pair id."""
    if not suffix or not isinstance(suffix, str):
        raise ValueError(f"CFG uncond request-id suffix must be a non-empty string, got {suffix!r}")
    _UNCOND_REQUEST_ID_SUFFIXES.add(suffix)


def cfg_uncond_suffixes() -> frozenset[str]:
    """Companion request-id suffixes registered in this process."""
    return frozenset(_UNCOND_REQUEST_ID_SUFFIXES)


def _derive_pair_id(request_id: str, role: str) -> str:
    """Pair id for a request that carries ``cfg_role`` but no ``cfg_pair_id``.

    The cond request id IS the pair id; its companion carries the same id plus
    a registered suffix, so stripping the suffix recovers it. Longest suffix
    first, so a registered suffix extending another still resolves.
    """
    if role == _UNCOND:
        for suffix in sorted(_UNCOND_REQUEST_ID_SUFFIXES, key=len, reverse=True):
            if request_id.endswith(suffix):
                return request_id[: -len(suffix)]
    return request_id


def _cfg_extra(request: Any, key: str) -> Any:
    sampling_params = getattr(request, "sampling_params", None)
    extra_args = getattr(sampling_params, "extra_args", None)
    if extra_args:
        return extra_args.get(key)
    return None


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
        # waiting; hold this member until the partner is schedulable.
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
    consumed next step, so nothing leaks.
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


def _patch_scheduler(scheduler_cls: type | None = None) -> None:
    """Wrap the scheduler methods that keep a CFG pair step-locked.

    ``scheduler_cls`` defaults to vLLM's v1 ``Scheduler``; accepting a class
    keeps the wrappers exercisable without building a real engine.
    """
    if scheduler_cls is None:
        from vllm.v1.core.sched.scheduler import Scheduler

        scheduler_cls = Scheduler

    orig_init = scheduler_cls.__init__
    orig_add = scheduler_cls.add_request
    orig_finish = scheduler_cls.finish_requests
    orig_schedule = scheduler_cls.schedule

    def _init(self, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        self._cfg_pairs: dict[str, dict[str, str]] = {}
        self._cfg_req_to_pair: dict[str, str] = {}

    def _add(self, request: Any) -> None:
        orig_add(self, request)
        pair_id = _cfg_extra(request, "cfg_pair_id")
        role = _cfg_extra(request, "cfg_role")
        if role and not pair_id:
            # Models that cannot inject a pair id from the serving layer send
            # the role alone and let the request id carry the pairing. It has
            # to be the user-facing id: the engine renames every request to an
            # internal UUID on admission, and the companion suffix survives
            # only on ``external_req_id``. Deriving from the UUID would give
            # the two members unrelated pair ids, so they would never be held
            # for each other and could be scheduled apart.
            external_id = getattr(request, "external_req_id", None) or request.request_id
            pair_id = _derive_pair_id(str(external_id), role)
        if pair_id and role:
            self._cfg_pairs.setdefault(pair_id, {})[role] = request.request_id
            self._cfg_req_to_pair[request.request_id] = pair_id
            # An asymmetric prefix-cache hit (the cond prompt shares cached
            # branches; the null prompt does not) can advance one member past
            # what post-schedule equalization can claw back. Skip prefix-cache
            # reads for CFG requests entirely.
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

    _schedule._cfg_pairing_patched = True  # type: ignore[attr-defined]
    # Legacy marker name kept for callers written against the Audex-only patch.
    _schedule._audex_cfg_patched = True  # type: ignore[attr-defined]

    scheduler_cls.__init__ = _init
    scheduler_cls.add_request = _add
    scheduler_cls.finish_requests = _finish
    scheduler_cls.schedule = _schedule


def apply_cfg_patches() -> None:
    """Make the vLLM v1 scheduler CFG-pair-aware. Idempotent, per process.

    Must run in the engine-core process before ``Scheduler`` is constructed
    (the ``__init__`` wrapper installs the pair registry).
    """
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True
    logger.info("Applying CFG pairing scheduler patches to vLLM v1")
    _patch_scheduler()
