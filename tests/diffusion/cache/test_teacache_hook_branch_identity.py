# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""
Regression test for TeaCacheHook branch-identity handling.

Boogu-Image's double-guidance path issues three transformer calls per
denoising step (cond+ref, neg+ref, neg+no-ref). The reference latents change
the joint token length, so the neg+ref and neg+no-ref calls are differently
shaped even though both are "negative" branches. Before ``cache_branch_hint``
was added, ``TeaCacheHook`` only alternated between two identities
("positive"/"negative") via a forward counter, so the third call silently
reused a residual cached for a different sequence length.

This exercises the hook directly (not just an extractor), through the real
cache-decision and residual-reuse path described in the PR review comment.
"""

from unittest.mock import patch

import pytest
import torch

from vllm_omni.diffusion.cache.teacache.config import TeaCacheConfig
from vllm_omni.diffusion.cache.teacache.extractors import CacheContext
from vllm_omni.diffusion.cache.teacache.hook import TeaCacheHook

pytestmark = [pytest.mark.cpu, pytest.mark.core_model]


def _make_extractor(seq_len: int):
    """Build a fake extractor whose transformer blocks just add a constant."""

    def extractor(module, hidden_states, **kwargs):
        del kwargs

        def run_transformer_blocks():
            return (hidden_states + 1.0,)

        return CacheContext(
            modulated_input=hidden_states,
            hidden_states=hidden_states,
            encoder_hidden_states=None,
            temb=torch.zeros(1),
            run_transformer_blocks=run_transformer_blocks,
            postprocess=lambda out: out,
        )

    return extractor


class _FakeModule(torch.nn.Module):
    """Stands in for the transformer; only needs attribute storage."""


def test_three_branch_guidance_keeps_states_separate_with_hint():
    """cache_branch_hint must isolate cond+ref / neg+ref / neg+no-ref state.

    Without the fix, ``neg+no-ref`` (shorter sequence, no reference tokens)
    would alias the ``cond+ref`` state under the 2-way alternation and the
    residual-reuse add would break on shape mismatch.
    """
    module = _FakeModule()
    module.do_true_cfg = True

    long_seq = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)  # with ref
    short_seq = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)  # no ref

    hook = TeaCacheHook(
        TeaCacheConfig(transformer_type="_Fake", rel_l1_thresh=100.0, coefficients=[0.0, 0.0, 0.0, 0.0, 1.0])
    )

    def dispatch(module, hidden_states, cache_branch=None):
        module.cache_branch_hint = cache_branch
        extractor = _make_extractor(hidden_states.shape[1])
        hook.extractor_fn = extractor
        return hook.new_forward(module, hidden_states=hidden_states)

    for _ in range(2):
        out_pos = dispatch(module, long_seq, cache_branch="positive")
        out_neg_ref = dispatch(module, long_seq, cache_branch="negative_ref")
        out_neg_noref = dispatch(module, short_seq, cache_branch="negative_noref")

    assert out_pos.shape == long_seq.shape
    assert out_neg_ref.shape == long_seq.shape
    assert out_neg_noref.shape == short_seq.shape

    pos_state = hook.state_manager._states["teacache_positive"]
    neg_ref_state = hook.state_manager._states["teacache_negative_ref"]
    neg_noref_state = hook.state_manager._states["teacache_negative_noref"]

    assert pos_state.previous_residual.shape == long_seq.shape
    assert neg_ref_state.previous_residual.shape == long_seq.shape
    assert neg_noref_state.previous_residual.shape == short_seq.shape


def test_without_hint_two_branch_alternation_aliases_third_call():
    """Documents the pre-fix failure: 2-way alternation collides on the 3rd call.

    With no ``cache_branch_hint``, the hook falls back to alternating
    positive/negative by forward count. The third (no-ref) call in a
    cond+ref, neg+ref, neg+no-ref sequence lands back on "positive" and
    reuses the cond+ref residual, which has the wrong shape once the
    no-ref call switches to a shorter sequence and a cache hit is forced.
    """
    module = _FakeModule()
    module.do_true_cfg = True
    module.cache_branch_hint = None

    long_seq = torch.zeros(1, 8, 1)
    short_seq = torch.zeros(1, 4, 1)

    # Force every decision after the first call to reuse the cache.
    hook = TeaCacheHook(
        TeaCacheConfig(transformer_type="_Fake", rel_l1_thresh=1e9, coefficients=[0.0, 0.0, 0.0, 0.0, 1.0])
    )

    def dispatch(hidden_states):
        hook.extractor_fn = _make_extractor(hidden_states.shape[1])
        return hook.new_forward(module, hidden_states=hidden_states)

    with patch(
        "vllm_omni.diffusion.cache.teacache.hook.get_classifier_free_guidance_world_size",
        return_value=1,
    ):
        dispatch(long_seq)  # forward_cnt 0 -> "positive", computed (state.cnt==0)
        dispatch(long_seq)  # forward_cnt 1 -> "negative", computed (state.cnt==0)
        with pytest.raises(RuntimeError):
            # forward_cnt 2 -> "positive" again: reuses the cond+ref residual
            # (shape [1, 8, 1]) against a [1, 4, 1] no-ref hidden state.
            dispatch(short_seq)
