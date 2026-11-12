# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import numpy as np
import pytest
import torch
from torch.nn import functional as F
from torch.nn.attention.flex_attention import create_block_mask

from tests.diffusion.ar_diffusion.test_segmented_kv import append, make_cache, make_state, rows
from vllm_omni.diffusion.models.hy_nomas.attention import RangeAttention, assemble_kv, build_range_mask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@torch.no_grad()
@pytest.mark.parametrize("restricted", [False, True])
def test_range_attention_matches_dense_math(restricted):
    torch.manual_seed(9)
    q, k, v = torch.randn(1, 4, 2, 8), torch.randn(1, 6, 2, 8), torch.randn(1, 6, 2, 8)
    qm, km = torch.tensor([[True, False, True, True]]), torch.tensor([[True, False, True, True, False, True]])
    ranges = [[(0, 2, 0, 3), (2, 4, 3, 6)]] if restricted else None
    mask = qm[:, :, None] & km[:, None, :]
    if restricted:
        mask[:, :2, 3:] = False
        mask[:, 2:, :3] = False
    actual = RangeAttention(8**-0.5)(q, k, v, ranges=ranges, query_mask=qm, key_mask=km)
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask[:, None]
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(actual[:, 1]) == 0


def test_range_mask_exact_rectangles_and_padding():
    mask = build_range_mask(
        3, 4, [(0, 2, 0, 2), (2, 3, 2, 4)], "cpu", np.array([True, False, True]), np.array([True, True, False, True])
    )
    actual = mask.mask_mod(torch.tensor(0), torch.tensor(0), torch.arange(4)[:, None], torch.arange(5)[None])
    expected = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    assert torch.equal(actual, expected)


def test_range_mask_rejects_overlapping_rectangles():
    with pytest.raises(ValueError, match="must not overlap"):
        build_range_mask(3, 4, [(0, 2, 0, 2), (1, 3, 1, 3)], "cpu", np.ones(3, dtype=bool), np.ones(4, dtype=bool))


@torch.no_grad()
def test_prebuilt_block_mask_preserves_compact_gqa():
    q, k, v = torch.randn(1, 4, 4, 8), torch.randn(1, 4, 2, 8), torch.randn(1, 4, 2, 8)

    def predicate(batch, head, query, key):
        return query // 2 == key // 2

    block_mask = create_block_mask(predicate, None, None, 4, 4, device="cpu")
    actual = RangeAttention(8**-0.5)(q, k, v, block_mask=block_mask)
    mask = (torch.arange(4)[:, None] // 2 == torch.arange(4)[None] // 2)[None, None]
    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2),
        k.repeat_interleave(2, 2).transpose(1, 2),
        v.repeat_interleave(2, 2).transpose(1, 2),
        attn_mask=mask,
    ).transpose(1, 2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


@torch.no_grad()
def test_resolve_masks_and_block_masks_bit_exact_parity_and_lru(mocker):
    import vllm_omni.diffusion.models.hy_nomas.attention as attn_mod

    assert getattr(RangeAttention.forward, "_torchdynamo_disable", False) is True
    torch.manual_seed(17)
    attn = RangeAttention(8**-0.5)
    q, k, v = torch.randn(2, 4, 4, 8), torch.randn(2, 6, 2, 8), torch.randn(2, 6, 2, 8)
    qm = torch.tensor([[True, False, True, True], [True, True, False, True]])
    km = torch.tensor([[True, False, True, True, False, True], [False, True, True, False, True, True]])
    ranges = [[(0, 2, 0, 3), (2, 4, 3, 6)], [(0, 3, 0, 4), (3, 4, 4, 6)]]

    expected = attn(q, k, v, ranges=ranges, query_mask=qm, key_mask=km)
    assert len(attn.masks) == 2

    spy_build = mocker.spy(attn_mod, "build_range_mask")
    resolved = attn.resolve_masks(
        q.shape[1], k.shape[1], q.device, batch_size=2, ranges=ranges, query_mask=qm, key_mask=km
    )
    assert isinstance(resolved, tuple) and len(resolved) == 2 and spy_build.call_count == 0

    actual = attn(q, k, v, block_masks=resolved)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    compiled_attn = torch.compile(attn, backend="aot_eager", dynamic=True)
    torch.testing.assert_close(compiled_attn(q, k, v, block_masks=resolved), expected, rtol=0, atol=0)
    assert spy_build.call_count == 0

    empty_resolved = attn.resolve_masks(0, 6, q.device, batch_size=2)
    assert empty_resolved == (None, None)
    assert attn(torch.empty(2, 0, 4, 8), k, v, block_masks=empty_resolved).shape == (2, 0, 4, 8)

    hot_key, evicted_key = list(attn.masks.keys())
    attn.resolve_masks(
        q.shape[1], k.shape[1], q.device, batch_size=1, ranges=ranges[:1], query_mask=qm[:1], key_mask=km[:1]
    )
    assert list(attn.masks.keys())[-1] == hot_key
    for extra_q in range(5, 36):
        attn.resolve_masks(extra_q, k.shape[1], q.device, batch_size=1)
    assert len(attn.masks) == 32 and hot_key in attn.masks and evicted_key not in attn.masks


@torch.no_grad()
def test_block_masks_rejects_conflicting_or_mismatched_arguments():
    attn = RangeAttention(8**-0.5)
    q, k, v = torch.randn(2, 4, 2, 8), torch.randn(2, 4, 2, 8), torch.randn(2, 4, 2, 8)
    masks = attn.resolve_masks(4, 4, q.device, batch_size=2)
    for ql, bs, kw in (
        (4, 1, {"query_mask": torch.ones(1, 4, dtype=torch.int64)}),
        (4, 2, {"key_mask": torch.ones(1, 4, dtype=torch.bool)}),
        (0, 1, {"query_mask": torch.ones(1, 4, dtype=torch.bool)}),
    ):
        with pytest.raises(ValueError, match="Boolean visibility masks"):
            attn.resolve_masks(ql, 4, q.device, batch_size=bs, **kw)
    for q_in, k_in, v_in, kw in (
        (q, k, v, {"block_masks": masks, "ranges": [[(0, 4, 0, 4)], [(0, 4, 0, 4)]]}),
        (q, k, v, {"block_masks": masks, "query_mask": torch.ones(2, 4, dtype=torch.bool)}),
        (q, k, v, {"block_masks": masks, "key_mask": torch.ones(2, 4, dtype=torch.bool)}),
        (q[:1], k[:1], v[:1], {"block_mask": masks[0], "block_masks": masks[:1]}),
        (q, k, v, {"block_masks": masks[:1]}),
        (q[:1], k[:1], v[:1], {"block_masks": ("invalid",)}),
        (q[:1], k[:1], v[:1], {"block_masks": masks[0]}),
        (torch.randn(2, 6, 2, 8), k, v, {"block_masks": masks}),
        (torch.empty(2, 0, 2, 8), k, v, {"block_masks": masks}),
    ):
        with pytest.raises(ValueError, match="Prebuilt Flex BlockMask"):
            attn(q_in, k_in, v_in, **kw)


@pytest.mark.parametrize("cache_text", [False, True])
@pytest.mark.parametrize("update", [False, True])
def test_contiguous_kv_preserves_history_and_condition_order(cache_text, update):
    keys, values = torch.zeros(1, 16, 2, 8), torch.zeros(1, 16, 2, 8)
    history = torch.randn(1, 3, 2, 8)
    keys[:, :3], values[:, :3] = history, -history
    current = torch.randn(1, 5, 2, 8)
    full_key, full_value = assemble_kv(
        {"k": keys, "v": values},
        0,
        current,
        -current,
        history=3,
        media_tokens=4,
        cache_text=cache_text,
        update=update,
    )
    count = 5 if cache_text or update else 4
    torch.testing.assert_close(full_key, torch.cat((history, current[:, :count]), 1), rtol=0, atol=0)
    torch.testing.assert_close(full_value, -full_key, rtol=0, atol=0)
    assert torch.equal(keys[:, :3], history)
    if cache_text or not update:
        assert full_key.untyped_storage().data_ptr() == keys.untyped_storage().data_ptr()


@pytest.mark.parametrize("cache_text", [False, True])
def test_segmented_kv_keeps_publication_with_its_owner(cache_text):
    cache = make_cache()
    owner = make_state(cache)
    state = owner.segmented("main")
    old = append(state, (3, 5), offset=7)
    current = rows(5, 31)[None]
    media = 5 if cache_text else 4
    state.prepare_layer_views(media, 0 if cache_text else 1)
    pending = state.begin_commit((media,))
    for layer in range(cache.num_layers):
        actual_key, actual_value = assemble_kv(
            state, layer, current, -current, history=8, media_tokens=4, cache_text=cache_text, update=True
        )
        expected = torch.cat((rows(8, 7 + 100 * layer)[None], current), 1)
        torch.testing.assert_close(actual_key, expected)
        torch.testing.assert_close(actual_value, -expected)
        assert actual_key.untyped_storage().data_ptr() == state._staging().layers[layer][0].untyped_storage().data_ptr()
        state.stage(layer)
    assert state.handles == old and state.history_tokens == 8
    state.commit(retain=(*old, *pending))
    assert state.history_tokens == 8 + media
    owner.close()
