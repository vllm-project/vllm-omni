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
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request

logger = init_logger(__name__)

_NORMAL_FINISH = (RequestStatus.FINISHED_STOPPED, RequestStatus.FINISHED_LENGTH_CAPPED)
_LOG_INTERVAL_S = 30.0


class PreemptionReserve:
    """Expected KV need of the admitted requests of one stage, in blocks."""

    def __init__(self, block_size: int, max_model_len: int, req_to_blocks: dict[str, list[KVCacheBlock]]):
        self.block_size = block_size
        self.max_model_len = max_model_len
        # The manager's own block lists: what a request holds, including blocks
        # allocated earlier in the current step (``num_computed_tokens`` lags them).
        self.req_to_blocks = req_to_blocks
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

    def reserved_blocks(self) -> int:
        """Blocks the admitted requests may still need beyond what they hold."""
        return sum(
            max(0, self.final_blocks(req, at) - len(self.req_to_blocks.get(rid, ())))
            for rid, (req, at) in self.admitted.items()
        )

    def on_free(self, request: Request) -> None:
        entry = self.admitted.pop(request.request_id, None)
        if entry is not None and request.status in _NORMAL_FINISH:
            grown = request.num_tokens - entry[1]
            self.max_growth = grown if self.max_growth is None else max(self.max_growth, grown)

    def log_held(self, reserved: int, own: int) -> None:
        now = time.monotonic()
        if now - self._last_log >= _LOG_INTERVAL_S:
            self._last_log = now
            logger.info(
                "Preemption reserve: admission held; %d admitted request(s) may still need %d block(s), "
                "the next waiting request %d more",
                len(self.admitted),
                reserved,
                own,
            )


def install_preemption_reserve(kv_cache_manager: KVCacheManager) -> PreemptionReserve | None:
    """Wrap ``kv_cache_manager.allocate_slots`` on this instance; return the bookkeeping object.

    Only for a single full-attention KV cache group, where one token takes one
    slot of one block; otherwise the block estimate would be wrong, so nothing
    is installed.
    """
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    managers = getattr(getattr(kv_cache_manager, "coordinator", None), "single_type_managers", None)
    # Exact type, as in fast_allocate: subclasses (sliding, circular-buffer, sink, ...) bound KV differently.
    if managers is None or len(managers) != 1 or type(managers[0]) is not FullAttentionManager:
        logger.warning("Preemption reserve not enabled: it supports a single full-attention KV cache group only")
        return None
    reserve = PreemptionReserve(
        int(managers[0].block_size), int(kv_cache_manager.max_model_len), managers[0].req_to_blocks
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
        rid = request.request_id
        # Admitted requests, resumable sessions and the first admission into an empty stage pass through.
        held_back = not (rid in reserve.admitted or request.resumable or not reserve.admitted)
        reserved = own = 0
        if held_back:
            covered = (
                request.num_computed_tokens
                + num_new_computed_tokens
                + num_external_computed_tokens
                + num_new_tokens
                + num_lookahead_tokens
            )
            reserved = reserve.reserved_blocks()
            own = max(0, reserve.final_blocks(request, request.num_tokens) - reserve.blocks(covered))
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
            reserved_blocks=reserved_blocks + reserved + own,
            has_scheduled_reqs=has_scheduled_reqs,
            skip_zeroing_group_ids=skip_zeroing_group_ids,
        )
        if result is None:
            if held_back:
                reserve.log_held(reserved, own)
        elif rid not in reserve.admitted and not request.resumable:
            reserve.admitted[rid] = (request, request.num_tokens)
        return result

    kv_cache_manager.allocate_slots = allocate_slots  # type: ignore[method-assign]
    logger.info("Preemption reserve enabled: admissions keep the expected growth of running requests in reserve")
    return reserve
