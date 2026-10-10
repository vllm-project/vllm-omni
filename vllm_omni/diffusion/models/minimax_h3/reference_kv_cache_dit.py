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
        if owner is not None and owner._reference_force_compute:
            return False
        return self._manager.can_cache(*args, **kwargs)


class MiniMaxH3CachedBlocks(CachedBlocks_Pattern_3_4_5):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._reference_force_compute = False
        self._reference_state = None
        self._reference_phase_key = None
        self._reference_skips_notified = False
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
        state = kwargs.get("reference_kv_tier1_state")
        if state is None:
            self._reference_phase_key = None
            return super().forward(hidden_states, *args, **kwargs)

        self.context_manager.set_context(self.cache_context)
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
        except Exception:
            self._reference_phase_key = None
            self._clear_reference_history()
            raise
        finally:
            self._reference_state = None
            self._reference_force_compute = False


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
