# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for the MagCache hook forward numerics.

The cached MagCache residual is the residual of the whole block stack
(tail output - head input). On a skipped step the head hook applies it
once; every other hooked block must pass its inputs through unchanged,
matching the reference implementation in diffusers.hooks.mag_cache.
"""

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.cache.magcache.config import MagCacheConfig
from vllm_omni.diffusion.cache.magcache.hook import (
    _MAG_CACHE_LEADER_BLOCK_HOOK,
    apply_mag_cache_hook,
)
from vllm_omni.diffusion.hooks.base import HookRegistry

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _CountingBlock(nn.Module):
    """Toy dual-stream block: adds a constant delta to hidden_states.

    Returns (encoder_hidden_states, hidden_states) to match the default
    TransformerBlockMetadata(return_hidden_states_index=1,
    return_encoder_hidden_states_index=0) registered by the hook.
    """

    def __init__(self, delta: float):
        super().__init__()
        self.delta = delta
        self.forward_calls = 0

    def forward(self, hidden_states, encoder_hidden_states):
        self.forward_calls += 1
        return encoder_hidden_states, hidden_states + self.delta


class _ToyTransformer(nn.Module):
    def __init__(self, n_blocks: int, delta: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList(_CountingBlock(delta) for _ in range(n_blocks))

    def forward(self, hidden_states, encoder_hidden_states):
        for block in self.blocks:
            encoder_hidden_states, hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states

    @property
    def total_forward_calls(self) -> int:
        return sum(block.forward_calls for block in self.blocks)


def _make_config(num_inference_steps: int = 2) -> MagCacheConfig:
    # retention_ratio=0.5 with 2 steps -> retention_step = 1: step 0 always
    # computes, step 1 with mag_ratio 1.0 accumulates zero error, which is a
    # guaranteed cache hit.
    return MagCacheConfig(
        transformer_type="Toy",
        threshold=0.24,
        max_skip_steps=5,
        retention_ratio=0.5,
        num_inference_steps=num_inference_steps,
        mag_ratios=torch.ones(num_inference_steps),
    )


def test_skipped_step_applies_full_stack_residual_once():
    n_blocks, delta = 3, 1.0
    model = _ToyTransformer(n_blocks=n_blocks, delta=delta)
    apply_mag_cache_hook(model, _make_config())

    x = torch.zeros(1, 4, 8)
    e = torch.zeros(1, 2, 8)

    # Step 0 computes and caches residual = tail_output - head_input.
    out_computed = model(x, e)
    assert torch.allclose(out_computed, x + n_blocks * delta)
    assert model.total_forward_calls == n_blocks

    # Step 1 is a cache hit: no block computes, and the cached residual is
    # applied exactly once, so the output matches the computed step.
    out_skipped = model(x, e)
    assert model.total_forward_calls == n_blocks, "step 1 was not skipped"
    assert torch.allclose(out_skipped, out_computed), (
        f"cached residual over-applied on skip: expected "
        f"{out_computed.flatten()[0].item()}, got {out_skipped.flatten()[0].item()}"
    )


def test_single_block_advances_schedule_on_skip():
    model = _ToyTransformer(n_blocks=1)
    apply_mag_cache_hook(model, _make_config(num_inference_steps=2))

    x = torch.zeros(1, 4, 8)
    e = torch.zeros(1, 2, 8)

    model(x, e)  # step 0: computed
    assert model.total_forward_calls == 1
    model(x, e)  # step 1: cache hit
    assert model.total_forward_calls == 1, "step 1 was not skipped"

    # The head hook is also the tail here, so a skipped step must still
    # advance the schedule; step 1 was the last step, which resets the
    # state for the next inference run.
    registry = HookRegistry.check_if_exists_or_initialize(model.blocks[0])
    hook = registry.get_hook(_MAG_CACHE_LEADER_BLOCK_HOOK)
    state = hook.state_manager.get_state()
    assert state.step_index == 0
    assert state._is_first_step
    assert state.previous_residual is None

    # A new run therefore starts by computing, not by replaying the stale
    # residual from the previous run.
    model(x, e)
    assert model.total_forward_calls == 2, "new run reused stale skip state"
