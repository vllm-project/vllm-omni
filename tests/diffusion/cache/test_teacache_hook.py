# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
Unit tests for TeaCacheHook CFG branch state management.

Regression tests for #2371: models with 3+ CFG branches (e.g. OmniGen2) need
one independent TeaCache state per branch. Before the fix the hook only
tracked two hardcoded states (positive/negative), so several branches shared
one state and corrupted each other's cache decisions. The fixed behavior:

- A branch id tagged on the transformer (``_teacache_branch_id``) selects a
  dedicated ``branch_{i}`` state and is consumed by exactly one forward.
- Untagged models keep the original 2-branch positive/negative behavior.
"""

from unittest.mock import patch

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

_HOOK = "vllm_omni.diffusion.cache.teacache.hook"


def _make_hook() -> tuple[TeaCacheHook, dict]:
    """Hook with identity rescaling (poly(x) = x) and a counting stub extractor."""
    config = TeaCacheConfig(rel_l1_thresh=0.2, coefficients=[0.0, 0.0, 0.0, 1.0, 0.0])
    hook = TeaCacheHook(config)
    compute_calls = {"n": 0}

    def stub_extractor(module, hidden_states, modulated_input, **kwargs):
        def run_transformer_blocks():
            compute_calls["n"] += 1
            return (hidden_states + 1.0,)

        return CacheContext(
            modulated_input=modulated_input,
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb=torch.zeros(1),
            run_transformer_blocks=run_transformer_blocks,
            postprocess=lambda h: h,
        )

    hook.extractor_fn = stub_extractor
    hook.state_manager.set_context("teacache")
    return hook, compute_calls


class TestTeaCacheHookMultiBranch:
    def test_tag_is_consumed_per_call(self):
        hook, _ = _make_hook()
        module = nn.Module()

        module._teacache_branch_id = 2
        hook.new_forward(module, hidden_states=torch.zeros(4), modulated_input=torch.ones(4))

        assert module._teacache_branch_id is None
        assert "teacache_branch_2" in hook.state_manager._states

    def test_each_tagged_branch_keeps_independent_state(self):
        """3 branches with distinct (but per-branch constant) modulated inputs.

        With per-branch state, step 2 sees zero rel-L1 distance inside every
        branch and reuses the cached residual (no recompute). With the old
        shared/polluted state, consecutive forwards belong to different
        branches, distances stay large and every step recomputes.
        """
        hook, compute_calls = _make_hook()
        module = nn.Module()
        branch_inputs = {0: torch.full((4,), 1.0), 1: torch.full((4,), 5.0), 2: torch.full((4,), 25.0)}

        for _step in range(2):
            for bid, modulated in branch_inputs.items():
                module._teacache_branch_id = bid
                hook.new_forward(module, hidden_states=torch.zeros(4), modulated_input=modulated)

        # One state per branch, each having seen exactly its own two forwards.
        assert set(hook.state_manager._states) == {
            "teacache_branch_0",
            "teacache_branch_1",
            "teacache_branch_2",
        }
        for bid, modulated in branch_inputs.items():
            hook.state_manager.set_context(f"teacache_branch_{bid}")
            state = hook.state_manager.get_state()
            assert state.cnt == 2
            assert torch.equal(state.previous_modulated_input, modulated)

        # Step 1 computed all 3 branches; step 2 hit the cache for all 3.
        assert compute_calls["n"] == 3

    def test_untagged_two_branch_behavior_unchanged(self):
        """Legacy models (no tag) must keep the original positive/negative states."""
        hook, _ = _make_hook()
        module = nn.Module()
        module.do_true_cfg = True

        with patch(f"{_HOOK}.get_classifier_free_guidance_world_size", return_value=1):
            for _ in range(4):
                hook.new_forward(module, hidden_states=torch.zeros(4), modulated_input=torch.ones(4))

        assert set(hook.state_manager._states) == {"teacache_positive", "teacache_negative"}

    def test_untagged_no_cfg_uses_positive_state(self):
        hook, _ = _make_hook()
        module = nn.Module()

        hook.new_forward(module, hidden_states=torch.zeros(4), modulated_input=torch.ones(4))

        assert set(hook.state_manager._states) == {"teacache_positive"}
