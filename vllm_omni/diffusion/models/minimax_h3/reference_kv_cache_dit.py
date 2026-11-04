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
        self._reference_phase_keys: dict[str, tuple] = {}
        self._primary_cache_context = self.cache_context
        self._phase_contexts_available = True
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

    def _activate_phase_context(self, phase: str) -> None:
        """Point cache_context at this phase's own context, creating it on first use.

        pattern_base re-activates ``self.cache_context`` inside its own forward, so the
        name has to be swapped on the instance rather than by calling set_context here.
        """
        name = self._primary_cache_context if phase == "refresh" else f"{self._primary_cache_context}::{phase}"
        manager = self.context_manager
        if name not in manager._cached_context_manager:
            base = manager._cached_context_manager[self._primary_cache_context]
            try:
                manager.set_context(name, *base._init_args, **base._init_kwargs)
            except Exception as exc:
                # Creation needs persistent_context; without it there is one context to
                # share, which is the pre-feature behaviour. Say so rather than guess.
                logger.warning_once(
                    "MiniMax-H3 reference-KV phase contexts unavailable, falling back to a "
                    "single shared cache context: %s",
                    exc,
                )
                self._phase_contexts_available = False
                self.cache_context = self._primary_cache_context
                return
        self.cache_context = name

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs):
        state = kwargs.get("reference_kv_state")
        if state is None:
            self._reference_phase_key = None
            self.cache_context = self._primary_cache_context
            return super().forward(hidden_states, *args, **kwargs)

        compact = bool(kwargs.get("reference_kv_compact", False))
        phase_contexts = bool(getattr(state, "phase_contexts", False)) and self._phase_contexts_available
        self.cache_context = self._primary_cache_context
        self.context_manager.set_context(self.cache_context)
        geometry = (
            ref(state),
            compact,
            tuple(hidden_states.shape),
            hidden_states.dtype,
            hidden_states.device,
        )
        if phase_contexts:
            # The epoch is deliberately not part of the key: a refresh updates the
            # reference KV the reuse phase reads, but reuse is built on that KV being
            # slowly varying, so its residual history stays valid across one.
            phase = "reuse" if compact else "refresh"
            changed = geometry != self._reference_phase_keys.get(phase)
            self._activate_phase_context(phase)
            self.context_manager.set_context(self.cache_context)
            self._reference_phase_keys[phase] = geometry
            phase_key = (phase, geometry)
        else:
            phase_key = (ref(state), state.reference_epoch, *geometry[1:])
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
            self._reference_phase_keys.clear()
            self._clear_reference_history()
            self.cache_context = self._primary_cache_context
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
