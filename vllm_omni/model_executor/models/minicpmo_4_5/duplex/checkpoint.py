# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Same-session Stage-0 KV checkpoints for MiniCPM-o duplex interruption."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from time import monotonic
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from vllm_omni.model_executor.models.minicpmo_4_5.duplex.policy import CHECKPOINT_KEY, REPLAY_CONTEXT_KEY
from vllm_omni.model_executor.models.minicpmo_4_5.duplex.window_kv import MiniCPMO45DuplexWindowManager

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request, StreamingUpdate

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

logger = init_logger(__name__)

_RETAIN_SECONDS = 15
_MAX_PREFIX_TOKENS = 3072
_CACHE_SHARE_DIVISOR = 32


def _duplex(value: object) -> dict[str, Any]:
    info = getattr(value, "model_intermediate_buffer", None)
    duplex = info.get("duplex") if isinstance(info, dict) else None
    return duplex if isinstance(duplex, dict) and duplex.get("data_plane") is True else {}


def _layout(duplex: dict[str, Any]) -> tuple[object, ...] | None:
    runtime = duplex["runtime_config"]
    mode = runtime.get("duplex_window_config", {}).get("sliding_window_mode", "off")
    if mode not in ("off", "basic"):
        return None
    return (
        mode,
        runtime.get("instructions"),
        runtime.get("initial_user_text"),
        runtime.get("duplex_first_append_context_tokens"),
        runtime.get("duplex_wrap_unit_audio_markers"),
        runtime.get("ref_audio_data"),
        runtime.get("ref_audio_format"),
        runtime.get("ref_audio_sample_rate_hz"),
    )


@dataclass
class _Boundary:
    session_id: str
    epoch: int
    history: str
    turn_id: int
    layout: tuple[object, ...]
    token_count: int
    terminator: int


@dataclass
class _Checkpoint:
    boundary: _Boundary
    blocks: list[KVCacheBlock]
    token_ids: list[int]
    last_sched_seq: int
    expires_at: float


class MiniCPMO45DuplexCheckpoints:
    def __init__(self, scheduler: OmniARScheduler):
        self._scheduler = scheduler
        cache = scheduler.kv_cache_manager
        managers = cache.coordinator.single_type_managers
        self._manager = (
            managers[0]
            if len(managers) == 1
            and type(managers[0]) in (FullAttentionManager, MiniCPMO45DuplexWindowManager)
            and not cache.enable_caching
            and scheduler.connector is None
            else None
        )
        self._pool = cache.block_pool
        self._checkpoints: dict[str, _Checkpoint] = {}

    def _release(self, session_id: str) -> None:
        saved = self._checkpoints.pop(session_id)
        self._pool.free_blocks(reversed(saved.blocks))

    def sweep(self) -> None:
        if not self._checkpoints:
            return
        now = monotonic()
        for session_id, saved in list(self._checkpoints.items()):
            if saved.expires_at <= now or self._pool.get_num_free_blocks() < len(saved.blocks):
                self._release(session_id)

    def record_boundary(self, request: Request, update: StreamingUpdate, confirmed_end: int) -> None:
        if self._manager is None:
            return
        incoming = _duplex(update)
        marker = incoming.get(CHECKPOINT_KEY)
        if not isinstance(marker, dict) or "completed_turn" not in marker:
            return
        previous = _duplex(request)
        layout = _layout(incoming)
        if (
            layout is None
            or not previous
            or _layout(previous) != layout
            or previous.get("turn_id") != marker.get("completed_turn")
            or not request.num_prompt_tokens <= confirmed_end < request.num_tokens
            or len(update.prompt_token_ids) <= 2
            or getattr(request, "_minicpmo_checkpoint_rebased", False)
        ):
            return
        # Stop at the last confirmed row. The sampled terminator and </unit>
        # close the unit in the new epoch, outside the retained KV.
        request._minicpmo_checkpoint_boundary = _Boundary(
            incoming["session_id"],
            incoming["epoch"],
            marker["history"],
            incoming["turn_id"],
            layout,
            confirmed_end,
            request._all_token_ids[confirmed_end],
        )

    def retain_before_abort(self, request: Request) -> None:
        manager = self._manager
        boundary = getattr(request, "_minicpmo_checkpoint_boundary", None)
        current = _duplex(request)
        if (
            manager is None
            or boundary is None
            or current.get("turn_id") != boundary.turn_id
            or _layout(current) != boundary.layout
            or boundary.token_count > _MAX_PREFIX_TOKENS
            or request.num_preemptions
            or boundary.token_count > request.num_computed_tokens - request.num_in_flight_tokens
        ):
            return
        self.sweep()
        num_blocks = ceil(boundary.token_count / manager.block_size)
        blocks = manager.req_to_blocks.get(request.request_id, ())[:num_blocks]
        if len(blocks) != num_blocks or any(block.is_null for block in blocks):
            return
        if boundary.session_id in self._checkpoints:
            self._release(boundary.session_id)
        retained_blocks = sum(len(saved.blocks) for saved in self._checkpoints.values())
        if retained_blocks + num_blocks > self._pool.num_gpu_blocks // _CACHE_SHARE_DIVISOR:
            return
        self._pool.touch(blocks)
        self._checkpoints[boundary.session_id] = _Checkpoint(
            boundary,
            list(blocks),
            list(request._all_token_ids[: boundary.token_count]),
            request.last_sched_seq,
            monotonic() + _RETAIN_SECONDS,
        )

    def restore(self, request: Request) -> None:
        manager = self._manager
        if manager is None:
            return
        incoming = _duplex(request)
        marker = incoming.get(CHECKPOINT_KEY)
        if not incoming or incoming.get("seq") != 1:
            return
        self.sweep()
        session_id = incoming["session_id"]
        saved = self._checkpoints.get(session_id)
        if saved is None:
            return
        # The old request may have written speculative positions sharing the
        # last retained page. Adopt only once all its scheduled writes drained.
        if (
            not isinstance(marker, dict)
            or incoming.get("epoch") != saved.boundary.epoch + 1
            or marker.get("history") != saved.boundary.history
            or _layout(incoming) != saved.boundary.layout
            or marker.get("interrupted_turn") != saved.boundary.turn_id
            or saved.last_sched_seq > self._scheduler.processed_step_seq
        ):
            self._release(session_id)
            return
        turns = incoming[REPLAY_CONTEXT_KEY]
        reused = marker["reuse_turns"]
        replay_budget = 2 + sum(turn["token_budget"] for turn in turns[reused:])
        prefix_length = saved.boundary.token_count
        context_tokens = incoming["runtime_config"].get("duplex_first_append_context_tokens")
        if not isinstance(context_tokens, int):
            self._release(session_id)
            return
        suffix_length = incoming["duplex_live_token_budget"] - context_tokens + replay_budget
        if (
            context_tokens >= prefix_length
            or suffix_length <= 0
            or prefix_length + suffix_length >= self._scheduler.max_model_len
        ):
            self._release(session_id)
            return
        self._checkpoints.pop(session_id)
        # Transfer the retained reference to this request's full-attention table.
        # The upstream APC admission API requires prefix caching to be enabled,
        # which is unsafe for the shared placeholder IDs in native duplex.
        manager.req_to_blocks[request.request_id] = saved.blocks
        manager.num_cached_block[request.request_id] = len(saved.blocks)
        request.prompt_token_ids = saved.token_ids + [incoming["scheduler_token_id"]] * suffix_length
        del request._all_token_ids[:]
        request._all_token_ids.extend(request.prompt_token_ids)
        request.num_computed_tokens = prefix_length
        request.num_prompt_tokens = len(request.prompt_token_ids)
        request._omni_has_native_checkpoint = True
        if saved.boundary.layout[0] == "basic":
            request._minicpmo45_window_units = [
                {"length": prefix_length - context_tokens, "generated_token_ids": []},
            ]
            request._minicpmo45_window_open_start = prefix_length
        incoming[CHECKPOINT_KEY] = {"reused_tokens": prefix_length, "terminator": saved.boundary.terminator}
        incoming[REPLAY_CONTEXT_KEY] = turns[reused:]
        incoming["duplex_replay_token_budget"] = replay_budget
        incoming["scheduler_token_budget"] = request.num_prompt_tokens
        logger.info("MiniCPM-o duplex re-used %d committed Stage-0 KV tokens", prefix_length)
