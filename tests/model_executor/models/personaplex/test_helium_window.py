# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Helium's vLLM sliding window attends to the keys Moshi's streaming ring keeps."""

import pytest
import torch

from vllm_omni.model_executor.models.personaplex.modeling_helium import moshi_attention_window
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
