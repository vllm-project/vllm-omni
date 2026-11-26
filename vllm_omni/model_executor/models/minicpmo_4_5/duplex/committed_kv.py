# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Bounded references to settled, complete Stage-0 KV blocks."""

from dataclasses import dataclass
from time import monotonic
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

from .policy import CHECKPOINT_KEY, REPLAY_CONTEXT_KEY

if TYPE_CHECKING:
    from vllm.v1.core.kv_cache_utils import KVCacheBlock
    from vllm.v1.request import Request, StreamingUpdate

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler

logger = init_logger(__name__)
_RETAIN_SECONDS = 15
_MAX_PREFIX_TOKENS = 3072
_CACHE_SHARE_DIVISOR = 32


def _duplex(request):
    info = getattr(request, "model_intermediate_buffer", None)
    duplex = info.get("duplex") if isinstance(info, dict) else None
    return duplex if isinstance(duplex, dict) and duplex.get("data_plane") is True else {}


def _layout(duplex):
    runtime = duplex["runtime_config"]
    if runtime.get("duplex_window_config", {}).get("sliding_window_mode") != "basic":
        return None
    return tuple(
        runtime.get(key)
        for key in (
            "instructions",
            "initial_user_text",
            "duplex_first_append_context_tokens",
            "duplex_wrap_unit_audio_markers",
            "ref_audio_data",
            "ref_audio_format",
            "ref_audio_sample_rate_hz",
        )
    )


@dataclass(frozen=True)
class _Boundary:
    session_id: str
    request_id: str
    history: str
    turn_id: int
    layout: tuple[object, ...]
    end: int
    terminator: int


@dataclass(frozen=True)
class _Saved:
    boundary: _Boundary
    blocks: tuple["KVCacheBlock", ...]
    last_step: int
    expires_at: float


class MiniCPMO45CommittedKV:
    def __init__(self, scheduler: "OmniARScheduler") -> None:
        from .window_kv import MiniCPMO45DuplexWindowManager

        self.scheduler = scheduler
        cache = scheduler.kv_cache_manager
        managers = cache.coordinator.single_type_managers
        self.manager = (
            managers[0]
            if len(managers) == 1
            and type(managers[0]) in (FullAttentionManager, MiniCPMO45DuplexWindowManager)
            and not cache.enable_caching
            and scheduler.connector is None
            else None
        )
        self.pool = cache.block_pool
        self.saved: dict[str, _Saved] = {}

    def release(self, session_id: str) -> None:
        saved = self.saved.pop(session_id, None)
        if saved is not None:
            self.pool.free_blocks(reversed(saved.blocks))

    def sweep(self) -> None:
        now = monotonic()
        for session_id, saved in tuple(self.saved.items()):
            if saved.expires_at <= now or self.pool.get_num_free_blocks() < len(saved.blocks):
                self.release(session_id)

    def record(self, request: "Request", update: "StreamingUpdate", settled_end: int) -> None:
        incoming = _duplex(update)
        marker = incoming.get(CHECKPOINT_KEY)
        if self.manager is None or not isinstance(marker, dict) or "completed_turn" not in marker:
            return
        previous = _duplex(request)
        layout = _layout(incoming)
        if (
            not previous
            or previous.get("epoch") != 0
            or previous.get("session_window_owned") is not True
            or previous.get("turn_id") != marker["completed_turn"]
            or layout is None
            or layout != _layout(previous)
            or getattr(request, "_minicpmo45_checkpoint_rebased", False)
            or getattr(request, "num_preemptions", 0)
            or not request.num_prompt_tokens <= settled_end < request.num_tokens
        ):
            return
        request._minicpmo45_committed_boundary = _Boundary(
            incoming["session_id"],
            request.request_id,
            marker["history"],
            incoming["turn_id"],
            layout,
            settled_end,
            request._all_token_ids[settled_end],
        )

    def retain(self, request: "Request") -> None:
        manager = self.manager
        boundary = getattr(request, "_minicpmo45_committed_boundary", None)
        duplex = _duplex(request)
        if (
            manager is None
            or boundary is None
            or duplex.get("turn_id") != boundary.turn_id
            or getattr(request, "_minicpmo45_checkpoint_rebased", False)
            or _layout(duplex) != boundary.layout
            or request.num_preemptions
            or boundary.end > min(_MAX_PREFIX_TOKENS, request.num_computed_tokens - request.num_in_flight_tokens)
        ):
            return
        self.sweep()
        # The remaining rows are recomputed. A partial page is never shared with old output.
        count = boundary.end // manager.block_size
        blocks = tuple(manager.req_to_blocks.get(request.request_id, ())[:count])
        if not blocks or len(blocks) != count or any(block.is_null for block in blocks):
            return
        self.release(boundary.session_id)
        if (
            sum(len(item.blocks) for item in self.saved.values()) + count
            > self.pool.num_gpu_blocks // _CACHE_SHARE_DIVISOR
        ):
            return
        self.pool.touch(blocks)
        self.saved[boundary.session_id] = _Saved(
            boundary, blocks, request.last_sched_seq, monotonic() + _RETAIN_SECONDS
        )

    def _adopt(self, request: "Request", saved: _Saved) -> int:
        manager = self.manager
        if manager is None:
            return 0
        manager.req_to_blocks[request.request_id] = list(saved.blocks)
        manager.num_cached_block[request.request_id] = len(saved.blocks)
        reused = len(saved.blocks) * manager.block_size
        request.num_computed_tokens = reused
        return reused

    def prepare(self, request: "Request", max_prompt_len: int, source: str | None) -> None:
        duplex = _duplex(request)
        marker = duplex.get(CHECKPOINT_KEY)
        if duplex.get("seq") != 1 or duplex.get("epoch") != 1 or not isinstance(marker, dict):
            return
        self.sweep()
        session_id = duplex["session_id"]
        saved = self.saved.get(session_id)
        if saved is None:
            return
        boundary = saved.boundary
        turns = duplex.get(REPLAY_CONTEXT_KEY)
        if (
            source != boundary.request_id
            or marker.get("history") != boundary.history
            or marker.get("interrupted_turn") != boundary.turn_id
            or _layout(duplex) != boundary.layout
            or saved.last_step > self.scheduler.processed_step_seq
            or not isinstance(turns, list)
            or not turns
            or turns[-1].get("user_only") is not True
        ):
            self.release(session_id)
            return
        context_tokens = duplex["runtime_config"]["duplex_first_append_context_tokens"]
        interrupted = turns[-1]
        tail_budget = interrupted["token_budget"] + 2
        live_length = request.num_prompt_tokens - duplex["duplex_replay_token_budget"] - context_tokens
        length = boundary.end + tail_budget + live_length
        if boundary.end <= context_tokens or live_length <= 0 or length > max_prompt_len:
            self.release(session_id)
            return
        token_id = duplex["scheduler_token_id"]
        request.prompt_token_ids = [token_id] * length
        del request._all_token_ids[:]
        request._all_token_ids.extend(request.prompt_token_ids)
        request.num_prompt_tokens = length
        request.block_hashes.clear()
        request.update_block_hashes()
        self.saved.pop(session_id)
        reused = self._adopt(request, saved)
        if not reused:
            self.pool.free_blocks(reversed(saved.blocks))
        duplex[CHECKPOINT_KEY] = {
            "source": boundary.request_id,
            "rows": boundary.end,
            "terminator": boundary.terminator,
        }
        duplex[REPLAY_CONTEXT_KEY] = [interrupted]
        duplex["duplex_replay_token_budget"] = tail_budget
        # The first worker window includes the replay and the new model unit.
        request._minicpmo45_window_open_start = duplex["open_input_start"]
        logger.info(
            "MiniCPM-o committed recovery: %d KV rows reused, %d initial rows scheduled", reused, length - reused
        )
