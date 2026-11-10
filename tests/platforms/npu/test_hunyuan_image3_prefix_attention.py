# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare the Ascend image-prefix adapter with an explicit joint-full mask."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.nn.functional import scaled_dot_product_attention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Load the device-independent helper without importing accelerator plugins.
_PATH = Path(__file__).resolve().parents[3] / "vllm_omni/platforms/npu/models/hunyuan_image3.py"
_SPEC = importlib.util.spec_from_file_location("hy3_npu_image_prefix", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


@pytest.mark.parametrize("cached", [0, 5, 9])
def test_image_prefix_matches_masked_sdpa_with_paged_context(cached):
    generator = torch.Generator().manual_seed(7)
    query = torch.randn(10, 4, 8, generator=generator)[cached:]
    key = torch.randn(10, 2, 8, generator=generator)
    value = torch.randn(10, 2, 8, generator=generator)
    table = torch.tensor([[2, 0, 1]])
    key_cache = torch.full((3, 4, 2, 8), 1000.0)
    value_cache = key_cache.clone()
    for position in range(10):
        block = table[0, position // 4]
        key_cache[block, position % 4] = key[position]
        value_cache[block, position % 4] = value[position]
    original_key, original_value = key_cache.clone(), value_cache.clone()
    metadata = SimpleNamespace(
        actual_seq_lengths_q=[len(query)],
        seq_lens_list=[10],
        block_tables=table,
        mm_prefix_range={0: [(1, 3), (6, 8)]},
    )

    def attend(q, k, v, mask=None):
        k = k.repeat_interleave(2, dim=1)
        v = v.repeat_interleave(2, dim=1)
        return scaled_dot_product_attention(
            q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1), attn_mask=mask
        ).transpose(0, 1)

    key_positions = torch.arange(10)
    query_positions = torch.arange(cached, 10)
    causal = key_positions[None, :] <= query_positions[:, None]
    expected_mask = causal.clone()
    for start, end in metadata.mm_prefix_range[0]:
        same_image = (query_positions >= start) & (query_positions <= end)
        expected_mask |= same_image[:, None] & (key_positions[None, :] <= end)
    expected = attend(query, key, value, expected_mask)
    output = attend(query, key, value, causal)
    calls = []

    def fia(**kwargs):
        calls.append(kwargs["actual_seq_lengths_kv"][0])
        length = calls[-1]
        pages = kwargs["block_table"][0]
        k = kwargs["key"].reshape_as(key_cache)[pages].reshape(-1, 2, 8)[:length]
        v = kwargs["value"].reshape_as(value_cache)[pages].reshape(-1, 2, 8)[:length]
        return attend(kwargs["query"], k, v), None

    actual = _MODULE.correct_image_queries(
        query,
        key_cache,
        value_cache,
        metadata,
        output,
        num_heads=4,
        num_kv_heads=2,
        scale=8**-0.5,
        fia=fia,
    )
    assert actual is output
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    assert torch.equal(key_cache, original_key) and torch.equal(value_cache, original_value)
    assert calls == {0: [4, 9], 5: [9], 9: []}[cached]


def test_incomplete_bidirectional_image_is_rejected():
    metadata = SimpleNamespace(actual_seq_lengths_q=[7], seq_lens_list=[7], mm_prefix_range={0: [(1, 8)]})
    with pytest.raises(ValueError, match="incomplete bidirectional image"):
        list(_MODULE.image_segments(metadata))
