# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helium's vLLM sliding window attends to the keys Moshi's streaming ring keeps."""

import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

from vllm_omni.model_executor.models.personaplex.modeling_helium import (
    check_window_bounds_kv_cache,
    check_window_bounds_kv_caches,
    moshi_attention_window,
)
from vllm_omni.model_executor.models.personaplex.personaplex_temporal import _ringkv_positions

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _moshi_keys(query: int, context: int) -> set[int]:
    """Keys a streaming Moshi layer attends to at ``query``: one key is written
    per step into a ring of ``context`` cells, then masked as in the port."""
    _, positions = _ringkv_positions(torch.tensor([query]), 1, context, torch.ones(1, dtype=torch.bool))
    return {k for k in positions[0].tolist() if k >= 0 and 0 <= query - k < context}


def _vllm_keys(query: int, window: int) -> set[int]:
    """vLLM's sliding window counts the query itself (FlashAttention
    window_size=(window - 1, 0))."""
    return set(range(max(0, query - window + 1), query + 1))


@pytest.mark.parametrize("context", [4, 8, 3000])
def test_helium_window_matches_the_moshi_ring_before_and_after_it_wraps(context: int) -> None:
    window = moshi_attention_window(context)
    queries = sorted({0, 1, context - 2, context - 1, context, context + 1, 2 * context, 5 * context + 3})
    for query in queries:
        assert _vllm_keys(query, window) == _moshi_keys(query, context), query
    # Once the ring has wrapped, the oldest of the last ``context`` keys is out.
    assert len(_moshi_keys(5 * context + 3, context)) == context - 1


def _kv_spec(sliding_window: int | None) -> FullAttentionSpec | SlidingWindowSpec:
    layout = {"block_size": 16, "num_kv_heads": 2, "head_size": 32, "dtype": torch.float16}
    if sliding_window is None:
        return FullAttentionSpec(**layout)
    return SlidingWindowSpec(**layout, sliding_window=sliding_window)


def test_context_past_the_window_needs_a_sliding_window_kv_cache() -> None:
    """A call longer than the window frees its KV only under a SlidingWindowSpec
    of Helium's window, so any other resolved spec is refused at startup."""
    check_window_bounds_kv_cache(46_080, 3000, _kv_spec(moshi_attention_window(3000)))
    with pytest.raises(ValueError, match="resolved FullAttentionSpec"):
        check_window_bounds_kv_cache(46_080, 3000, _kv_spec(None))
    with pytest.raises(ValueError, match=r"resolved SlidingWindowSpec\(sliding_window=3000\)"):
        check_window_bounds_kv_cache(46_080, 3000, _kv_spec(3000))
    with pytest.raises(ValueError, match="resolved NoneType"):
        check_window_bounds_kv_cache(46_080, 3000, None)


def test_context_within_the_window_accepts_any_kv_cache() -> None:
    # No call can outgrow the window, so a backend without one is harmless.
    check_window_bounds_kv_cache(3000, 3000, _kv_spec(None))
    check_window_bounds_kv_cache(3000, 3000, None)


def test_a_rank_whose_layers_expose_no_attention_layer_is_refused() -> None:
    """Finding nothing to check must not pass for a check that did not run."""
    with pytest.raises(ValueError, match="none of this rank's 32 layers"):
        check_window_bounds_kv_caches(46_080, 3000, [], num_layers=32)
    # An empty pipeline-parallel slice holds no Helium KV, so there is nothing to check.
    check_window_bounds_kv_caches(46_080, 3000, [], num_layers=0)
