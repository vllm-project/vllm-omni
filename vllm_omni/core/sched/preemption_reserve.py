# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Hold back admissions that would likely force a preemption on a stage that cannot replay.

A stage declared with ``recompute_preemption: fail`` finishes a request with an
error when the scheduler has to preempt it (#9249). The upstream scheduler still
admits a waiting request as soon as its next prefill chunk fits, so under load
the admitted requests later outgrow the pool and the newest are dropped.

This wraps ``kv_cache_manager.allocate_slots`` so that a request which is not yet
admitted gets blocks only if the free pool also covers what every admitted
request is still expected to need (rest of its prompt and its expected growth)
plus its own. It reuses ``reserved_blocks``, the argument the upstream scheduler
passes for async KV loads to avoid predictable preemptions. Requests that do not
fit stay waiting.

Expected growth (tokens added after admission) is an estimate: the largest
growth seen on a request that finished normally, capped by the request's
``max_tokens``; ``max_tokens`` until a request finishes. A request that grows
more than that can still be preempted and fails as before. Not covered:
resumable streaming sessions (passed through untracked), and KV layouts other
than a single full-attention group (not installed).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from collections.abc import Callable

    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

_NORMAL_FINISH = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
_LOG_INTERVAL_S = 30.0


class PreemptionReserve:
    """Expected KV need of the admitted requests of one stage, in blocks."""

    def __init__(
        self,
        block_size: int,
        max_model_len: int,
        req_to_blocks: dict[str, list[KVCacheBlock]],
        scheduler_reserved: Callable[[Request], int],
    ):
        self.block_size = block_size
        self.max_model_len = max_model_len
        # The manager's own block lists: what a request holds, including blocks
        # allocated earlier in the current step (``num_computed_tokens`` lags them).
        self.req_to_blocks = req_to_blocks
        # Blocks the scheduler's own ``reserved_blocks`` already counts for a request.
        self.scheduler_reserved = scheduler_reserved
        # request_id -> (request, num_tokens when it was admitted)
        self.admitted: dict[str, tuple[Request, int]] = {}
        self.max_growth: int | None = None
        self._last_log = 0.0

    def blocks(self, tokens: int) -> int:
        return -(-max(0, tokens) // self.block_size)

    def growth(self, request: Request) -> int:
        return request.max_tokens if self.max_growth is None else min(request.max_tokens, self.max_growth)

    def final_blocks(self, request: Request, num_tokens_at_admission: int) -> int:
        """Blocks ``request`` is expected to hold when it finishes."""
        target = max(num_tokens_at_admission + self.growth(request), request.num_tokens)
        return self.blocks(min(target, self.max_model_len))

    def reserved_blocks(self, caller_reserved: int = 0) -> int:
        """Blocks the admitted requests may still need beyond what they hold, on top of ``caller_reserved``.

        For an async KV load the caller's reservation covers the unfinished prefill
        of every in-flight request (plus speculative-decode headroom). Requests that
        are also admitted here count once, at the larger of the two estimates; the
        rest of the caller's reservation stays on top.
        """
        total = overlap = 0
        for rid, (req, at) in self.admitted.items():
            need = max(0, self.final_blocks(req, at) - len(self.req_to_blocks.get(rid, ())))
            total += need
            if caller_reserved:
                overlap += min(need, self.scheduler_reserved(req))
        # Guard: a caller whose reservation does not cover these requests is never undercut.
        return total - min(overlap, caller_reserved)

    def on_free(self, request: Request) -> None:
        entry = self.admitted.pop(request.request_id, None)
        if entry is not None and request.status in _NORMAL_FINISH:
            grown = request.num_tokens - entry[1]
            self.max_growth = grown if self.max_growth is None else max(self.max_growth, grown)

    def admission_blocks(self, request: Request, covered_tokens: int, caller_reserved: int) -> int:
        """Blocks to reserve on top of ``caller_reserved`` before ``request`` gets blocks.

        Zero for requests that are not held back: admitted ones, resumable sessions
        and the first admission into an empty stage. ``covered_tokens`` are the
        tokens this allocation will cover.
        """
        if request.request_id in self.admitted or request.resumable or not self.admitted:
            return 0
        own = max(0, self.final_blocks(request, request.num_tokens) - self.blocks(covered_tokens))
        return self.reserved_blocks(caller_reserved) + own

    def record(self, request: Request, allocated: bool, held: int) -> None:
        """Track a newly admitted request, or log (rate-limited) that ``held`` blocks kept it waiting."""
        if allocated:
            if request.request_id not in self.admitted and not request.resumable:
                self.admitted[request.request_id] = (request, request.num_tokens)
            return
        now = time.monotonic()
        if held and now - self._last_log >= _LOG_INTERVAL_S:
            self._last_log = now
            logger.info(
                "Preemption reserve: admission held; %d admitted request(s) and the next waiting request "
                "may still need %d block(s) beyond the scheduler's own reservation",
                len(self.admitted),
                held,
            )


def install_preemption_reserve(
    kv_cache_manager: KVCacheManager,
    scheduler_reserved: Callable[[Request], int],
) -> PreemptionReserve | None:
    """Wrap ``kv_cache_manager.allocate_slots`` on this instance; return the bookkeeping object.

    ``scheduler_reserved(request)`` gives the blocks the caller's own
    ``reserved_blocks`` counts for ``request``, so a request reserved by both is
    counted once. Only for a single full-attention KV cache group, where one
    token takes one slot of one block; otherwise the block estimate would be
    wrong, so nothing is installed.
    """
    managers = getattr(getattr(kv_cache_manager, "coordinator", None), "single_type_managers", None)
    # Exact type, as in fast_allocate: subclasses (sliding, circular-buffer, sink, ...) bound KV differently.
    if managers is None or len(managers) != 1 or type(managers[0]) is not FullAttentionManager:
        logger.warning("Preemption reserve not enabled: it supports a single full-attention KV cache group only")
        return None
    reserve = PreemptionReserve(
        int(managers[0].block_size),
        int(kv_cache_manager.max_model_len),
        managers[0].req_to_blocks,
        scheduler_reserved,
    )
    original = kv_cache_manager.allocate_slots

    def allocate_slots(
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
        skip_zeroing_group_ids: tuple[int, ...] = (),
    ) -> KVCacheBlocks | None:
        covered = (
            request.num_computed_tokens
            + num_new_computed_tokens
            + num_external_computed_tokens
            + num_new_tokens
            + num_lookahead_tokens
        )
        held = reserve.admission_blocks(request, covered, reserved_blocks)
        result = original(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks + held,
            has_scheduled_reqs=has_scheduled_reqs,
            skip_zeroing_group_ids=skip_zeroing_group_ids,
        )
        reserve.record(request, result is not None, held)
        return result

    kv_cache_manager.allocate_slots = allocate_slots  # type: ignore[method-assign]
    logger.info("Preemption reserve enabled: admissions keep the expected growth of running requests in reserve")
    return reserve
