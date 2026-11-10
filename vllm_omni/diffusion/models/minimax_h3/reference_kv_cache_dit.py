# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Model-specific coordination of Cache-DiT and MiniMax-H3 reference KV."""

from collections.abc import Iterator
from typing import Any
from weakref import ref

import torch
import torch.nn as nn
from cache_dit import BlockAdapter
from cache_dit.caching.cache_adapters.cache_adapter import CachedAdapter
from cache_dit.caching.cache_blocks.pattern_3_4_5 import CachedBlocks_Pattern_3_4_5
from vllm.logger import init_logger

from .reference_kv_cache import MiniMaxH3ReferenceKVTier2State, ReferenceKVRefreshRequiredError

logger = init_logger(__name__)


def iter_minimax_h3_blocks(blocks: nn.ModuleList) -> Iterator[nn.Module]:
    """Inspect physical layers without bypassing Cache-DiT during execution."""
    for block in blocks:
        if hasattr(block, "attn"):
            yield block
        elif hasattr(block, "transformer_blocks"):
            yield from iter_minimax_h3_blocks(block.transformer_blocks)
        else:
            raise TypeError(f"Unsupported MiniMax-H3 block wrapper: {type(block).__name__}")


class _ReferenceCacheManager:
    """Delegate bookkeeping; veto hits on a reference refresh/layout change."""

    def __init__(self, manager: Any, owner: nn.Module) -> None:
        self._manager = manager
        self._owner = ref(owner)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._manager, name)

    def can_cache(self, *args, **kwargs) -> bool:
        owner = self._owner()
        state = owner._reference_state if owner is not None else None
        if owner is not None and owner._reference_force_compute:
            if state is not None and state.refresh_due:
                self._request_refresh(owner, state)
            return False
        hit = self._manager.can_cache(*args, **kwargs)
        if state is not None and state.refresh_due:
            if hit:
                state.refresh_reason = "cache_hit_deferred"
            else:
                self._request_refresh(owner, state)
        return hit

    @staticmethod
    def _request_refresh(owner: "MiniMaxH3CachedBlocks", state: MiniMaxH3ReferenceKVTier2State) -> None:
        # The clock has advanced, but no middle output/cache hit is committed.
        owner._reference_resume_key = (ref(state), state.current_step)
        raise ReferenceKVRefreshRequiredError("due reference KV on a Cache-DiT miss")

    def mark_step_begin(self) -> None:
        owner = self._owner()
        state = owner._reference_state if owner is not None else None
        key = (ref(state), state.current_step) if state is not None else None
        if key is not None and owner._reference_resume_key == key:
            owner._reference_resume_key = None
            # The probe already advanced the request clock. Full-layout
            # history has just been reset; initialize its calibrators once.
            context = self._manager.get_context()
            if context.has_calibrators():
                for calibrator in context.get_calibrators():
                    if calibrator is not None:
                        calibrator.mark_step_begin()
            return
        if owner is not None:
            owner._reference_resume_key = None
        self._manager.mark_step_begin()


class MiniMaxH3CachedBlocks(CachedBlocks_Pattern_3_4_5):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reference_force_compute = False
        self._reference_state = None
        self._reference_phase_key = None
        self._reference_skips_notified = False
        self._reference_resume_key = None
        self.context_manager = _ReferenceCacheManager(self.context_manager, self)

    def _clear_reference_history(self) -> None:
        context = self.context_manager.get_context()
        context.clear_buffers()
        if context.has_calibrators():
            for calibrator in (*context.get_calibrators(), *context.get_cfg_calibrators()):
                if calibrator is not None:
                    calibrator.reset_cache()
        # Keep the request step clock/hit list; restart accumulation for the
        # new KV version and hidden-state layout.
        context.continuous_cached_steps = 0
        context.cfg_continuous_cached_steps = 0
        context.accumulated_residual_diff = 0.0
        context.cfg_accumulated_residual_diff = 0.0

    def _notify_reference_skips(self) -> None:
        if self._reference_state is None or self._reference_skips_notified:
            return
        indices = [block.layer_index for block in iter_minimax_h3_blocks(self._Mn_blocks())]
        self._reference_state.mark_block_cache_skipped(indices)
        self._reference_skips_notified = True
        logger.info(
            "MiniMax-H3 Cache-DiT reference-KV hit: step=%d, skipped_blocks=%d",
            self._reference_state.current_step,
            len(indices),
        )

    def call_Bn_blocks(self, hidden_states: torch.Tensor, *args, **kwargs):
        if self._is_in_cache_step():
            # The tail needs its KV, not the next sequential middle layer.
            self._notify_reference_skips()
        return super().call_Bn_blocks(hidden_states, *args, **kwargs)

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        state = kwargs.get("reference_kv_state")
        if state is None:
            self._reference_phase_key = None
            return super().forward(hidden_states, *args, **kwargs)

        self.context_manager.set_context(self.cache_context)
        if state.current_step == 0 and state.cache_dit_policy == "hit":
            # Cache-DiT installs this wrapper only inside its patched forward.
            configure_reference_cache_schedule(nn.ModuleList([self]), state)
        phase_key = (
            ref(state),
            state.reference_epoch,
            bool(kwargs.get("reference_kv_compact", False)),
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
        )
        changed = phase_key != self._reference_phase_key
        self._reference_force_compute = (
            state.is_reference_refresh_step or changed or not state.supports_block_cache_reuse
        )
        if changed:
            self._clear_reference_history()
        self._reference_phase_key = phase_key
        self._reference_state = state
        self._reference_skips_notified = False
        try:
            result = super().forward(hidden_states, *args, **kwargs)
            if self._is_in_cache_step():
                # Bn=0 has no tail callback, but still must account for skips.
                self._notify_reference_skips()
            return result
        except ReferenceKVRefreshRequiredError:
            # Keep the single-use clock token for the full-layout replay.
            self._reference_phase_key = None
            raise
        except Exception:
            self._reference_resume_key = None
            self._reference_phase_key = None
            self._clear_reference_history()
            raise
        finally:
            self._reference_state = None
            self._reference_force_compute = False


def configure_reference_cache_schedule(blocks: nn.ModuleList, state: MiniMaxH3ReferenceKVTier2State | None) -> None:
    """Enable hit priority only for the supported single H3 wrapper contract."""
    if state is None or state.cache_dit_policy != "hit":
        return
    supported = (
        isinstance(state, MiniMaxH3ReferenceKVTier2State)
        and len(blocks) == 1
        and isinstance(blocks[0], MiniMaxH3CachedBlocks)
        and len(list(iter_minimax_h3_blocks(blocks))) == state.num_layers
    )
    if supported:
        wrapper = blocks[0]
        wrapper.context_manager.set_context(wrapper.cache_context)
        supported = not wrapper.context_manager.get_context().cache_config.enable_separate_cfg
    state.cache_dit_schedule_active = supported
    if supported and state.current_step == 0:
        state.refresh_reason = "initial"
    logger.info(
        "MiniMax-H3 reference-KV Cache-DiT schedule: requested=%s, effective=%s, "
        "refresh_interval=%d, max_refresh_interval=%s",
        state.cache_dit_policy,
        "hit" if supported else "refresh",
        state.refresh_interval,
        state.max_refresh_interval,
    )


class MiniMaxH3CachedAdapter(CachedAdapter):
    @classmethod
    def collect_unified_blocks(
        cls, block_adapter: BlockAdapter, contexts_kwargs: list[dict]
    ) -> list[dict[str, nn.ModuleList]]:
        BlockAdapter.assert_normalized(block_adapter)
        total = []
        context_index = 0
        for i, transformer in enumerate(block_adapter.transformer):
            contexts = {}
            for j, blocks in enumerate(block_adapter.blocks[i]):
                cache_config = contexts_kwargs[context_index]["cache_config"]
                context_index += 1
                contexts[block_adapter.unique_blocks_name[i][j]] = nn.ModuleList(
                    [
                        MiniMaxH3CachedBlocks(
                            blocks,
                            transformer=transformer,
                            forward_pattern=block_adapter.forward_pattern[i][j],
                            check_forward_pattern=block_adapter.check_forward_pattern,
                            check_num_outputs=block_adapter.check_num_outputs,
                            cache_prefix=block_adapter.blocks_name[i][j],
                            cache_context=block_adapter.unique_blocks_name[i][j],
                            context_manager=block_adapter.pipe._context_manager,
                            cache_type=cache_config.cache_type,
                        )
                    ]
                )
            total.append(contexts)
        logger.info("MiniMax-H3 Cache-DiT reference-KV coordination enabled")
        return total
