# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import hashlib
from functools import cache

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.models.magi2.configuration_magi2 import MAGI2_REFINER_CONFIG
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.window_attention import (
    Magi2WindowAttentionKernel,
    Magi2WindowAttentionPlan,
    Magi2WindowRanges,
    build_reference_window_ranges,
    build_window_attention_plan,
    torch_window_attention_reference,
    ulysses_window_attention,
    window_attention,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_PRODUCTION_GRID = (63, 68, 120)
_WINDOW = MAGI2_REFINER_CONFIG.window

# sha256[:16] of the int32 q_ranges, k_ranges, token_order and token_restore_order bytes produced by
# SandAI's Magi2RefinerDataProxy._build_window_local_attn_handler / _build_block_order, plus the row
# count and the number of dense rows.
_REFERENCE_LAYOUTS = {
    (_PRODUCTION_GRID, 512): (
        "a731527dde7bfbf9",
        "c89c0120dd441557",
        "9553ee706fd89a31",
        "a832fe5bba68b1bb",
        390872,
        4088,
    ),
    (_PRODUCTION_GRID, 1000): (
        "9d49b98cf9ae571c",
        "d164c75ba256ff86",
        "5189665ffb921f6a",
        "77b9b10015c53c5e",
        390880,
        4096,
    ),
    (_PRODUCTION_GRID, 0): ("ff4a63b1a1f9c718", "20dd70c5d06b7686", "4bf2f1e7b6f424ef", "4a1a96a0ad346a50", 386784, 0),
    ((27, 8, 24), 37): ("0f70805c0145a17b", "0abaf372cfe358fc", "3afecfe1c6674edb", "f7818d6c915e2552", 1394, 50),
    ((23, 16, 40), 0): ("3d2b79245a687974", "eb561c01ccd4a5a2", "c870f3c389037ebc", "ff0f187e92126045", 5544, 0),
    ((23, 16, 40), 5): ("af127dccfdbe635e", "c8975434370e220b", "14c919898db48e9f", "fb41b81b43335cb9", 5666, 122),
    # Partial h and w blocks: the reference layout is defined for any grid.
    ((27, 6, 22), 3): ("05ee4f5e45ef1166", "68787fccda9e67a2", "157ebf363115d27a", "e0a129ddc96250d6", 1394, 50),
}


def _sha16(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().numpy().tobytes()).hexdigest()[:16]


@cache
def _ranges(grid: tuple[int, int, int], text_len: int) -> Magi2WindowRanges:
    return build_reference_window_ranges(grid, text_len)


@cache
def _plan(grid: tuple[int, int, int], text_len: int) -> Magi2WindowAttentionPlan:
    return build_window_attention_plan(grid, text_len)


@pytest.mark.parametrize(("grid", "text_len"), list(_REFERENCE_LAYOUTS))
def test_reference_ranges_reproduce_the_sandai_layout(grid: tuple[int, int, int], text_len: int) -> None:
    q_hash, k_hash, order_hash, restore_hash, rows, dense_rows = _REFERENCE_LAYOUTS[grid, text_len]
    ranges = _ranges(grid, text_len)
    num_tokens = int(np.prod(grid)) + text_len

    for tensor in (ranges.q_ranges, ranges.k_ranges, ranges.token_order, ranges.token_restore_order):
        assert tensor.dtype == torch.int32
        assert tensor.is_contiguous()
    assert ranges.q_ranges.shape == ranges.k_ranges.shape == (rows, 2)
    assert ranges.token_order.shape == ranges.token_restore_order.shape == (num_tokens,)
    assert ranges.num_dense_rows == dense_rows
    assert (ranges.max_seqlen_q, ranges.max_seqlen_k) == (128, num_tokens)
    assert _sha16(ranges.q_ranges) == q_hash
    assert _sha16(ranges.k_ranges) == k_hash
    assert _sha16(ranges.token_order) == order_hash
    assert _sha16(ranges.token_restore_order) == restore_hash


def test_reference_ranges_production_structure() -> None:
    ranges = _ranges(_PRODUCTION_GRID, 512)
    num_video, num_tokens = 514_080, 514_592
    sizes = (ranges.block_ranges[:, 1] - ranges.block_ranges[:, 0]).tolist()

    assert ranges.block_grid == (8, 17, 30)
    assert (sizes.count(128), sizes.count(112), len(sizes)) == (3570, 510, 4080)
    # 4080 video rows + 2 x 4 text chunks, then one window row per in-grid (query, key) block pair.
    assert ranges.num_dense_rows == 4080 + 2 * 4
    assert ranges.q_ranges.shape[0] - ranges.num_dense_rows == 34 * 79 * 144
    assert ranges.q_ranges[:2].tolist() == [[0, 128], [128, 256]]
    assert ranges.k_ranges[:4080].unique(dim=0).tolist() == [[num_video, num_tokens]]
    assert ranges.q_ranges[4080:4088].tolist() == 2 * [
        [514080, 514208],
        [514208, 514336],
        [514336, 514464],
        [514464, 514592],
    ]
    assert ranges.k_ranges[4080:4088].tolist() == 4 * [[0, num_video]] + 4 * [[num_video, num_tokens]]
    # First window row: offset (-2, -2, -2) of the first query block that has that neighbour.
    first_window = ranges.num_dense_rows
    assert ranges.q_ranges[first_window].tolist() == [138496, 138624]
    assert ranges.k_ranges[first_window].tolist() == [0, 128]
    order = ranges.token_order.long()
    assert torch.equal(ranges.token_restore_order.long()[order], torch.arange(num_tokens))
    assert torch.equal(order[num_video:], torch.arange(num_video, num_tokens))


def test_reference_ranges_without_text_have_no_dense_rows() -> None:
    with_text = _ranges(_PRODUCTION_GRID, 512)
    without_text = _ranges(_PRODUCTION_GRID, 0)

    assert without_text.num_dense_rows == 0
    assert torch.equal(without_text.q_ranges, with_text.q_ranges[with_text.num_dense_rows :])
    assert torch.equal(without_text.k_ranges, with_text.k_ranges[with_text.num_dense_rows :])
    assert torch.equal(without_text.token_order, with_text.token_order[:514_080])


def _assert_plan_matches_ranges(plan: Magi2WindowAttentionPlan, ranges: Magi2WindowRanges) -> np.ndarray:
    """Assert the plan's key set per query row equals the union of the reference rows; return window rows per block."""

    page_size = plan.page_size
    num_video, num_tokens, text_len = ranges.num_video_tokens, ranges.num_tokens, ranges.text_len
    num_blocks = ranges.block_ranges.shape[0]
    starts = ranges.block_ranges[:, 0].long().numpy()
    ends = ranges.block_ranges[:, 1].long().numpy()
    q = ranges.q_ranges.long().numpy()
    k = ranges.k_ranges.long().numpy()

    assert np.array_equal(plan.cu_seqlens_q.long().numpy(), np.append(starts, num_video))
    assert plan.max_seqlen_q == int((ends - starts).max())

    # Reference video rows: each query range is one whole block.
    video_rows = q[:, 0] < num_video
    block = np.searchsorted(starts, q[video_rows, 0])
    assert np.array_equal(starts[block], q[video_rows, 0])
    assert np.array_equal(ends[block], q[video_rows, 1])
    video_k = k[video_rows]
    text_key_row = video_k[:, 0] == num_video
    assert np.all(video_k[text_key_row, 1] == num_tokens)
    assert np.array_equal(
        np.bincount(block[text_key_row], minlength=num_blocks), np.full(num_blocks, int(text_len > 0))
    )
    window_k = video_k[~text_key_row]
    window_block = block[~text_key_row]
    assert np.all(window_k[:, 1] <= num_video)
    assert np.all(window_k % page_size == 0)
    pages_per_row = (window_k[:, 1] - window_k[:, 0]) // page_size
    offsets = np.arange(pages_per_row.sum()) - np.repeat(np.cumsum(pages_per_row) - pages_per_row, pages_per_row)
    reference_rows = np.repeat(window_block, pages_per_row)
    reference_pages = np.repeat(window_k[:, 0] // page_size, pages_per_row) + offsets

    # Plan rows: the pages a kernel reads, i.e. the first ceil(seqused_k / page_size) entries.
    seqused = plan.seqused_k.long().numpy()
    page_table = plan.page_table.long().numpy()
    used_pages = -(-seqused // page_size)
    in_row = np.arange(page_table.shape[1])[None, :] < used_pages[:, None]
    plan_rows = np.nonzero(in_row)[0]
    plan_pages = page_table[in_row]
    is_video_page = plan_pages < num_video // page_size

    def keys(rows: np.ndarray, pages: np.ndarray) -> np.ndarray:
        return np.sort(rows * (plan.num_pages + 1) + pages)

    reference_keys = keys(reference_rows, reference_pages)
    assert np.all(np.diff(reference_keys) > 0), "reference window ranges overlap"
    assert np.array_equal(keys(plan_rows[is_video_page], plan_pages[is_video_page]), reference_keys)

    # Text pages follow the video pages in order, and seqused_k ends the row exactly at the last text token.
    video_pages = np.bincount(plan_rows[is_video_page], minlength=num_blocks)
    assert np.array_equal(seqused, video_pages * page_size + text_len)
    text_pages = -(-text_len // page_size)
    if text_pages:
        columns = video_pages[:, None] + np.arange(text_pages)[None, :]
        expected = np.broadcast_to(num_video // page_size + np.arange(text_pages), columns.shape)
        assert np.array_equal(np.take_along_axis(page_table, columns, axis=1), expected)
    assert plan.max_seqlen_k == int(seqused.max())

    # Text queries: every reference chunk sees [0, V) and [V, N); the plan's text row sees every token.
    text_q, text_k = q[~video_rows], k[~video_rows]
    chunks = np.unique(text_q, axis=0)
    assert np.array_equal(chunks[:, 0], np.arange(num_video, num_tokens, _WINDOW.dense_query_chunk))
    assert chunks.shape[0] == 0 or chunks[-1, 1] == num_tokens
    for chunk in chunks:
        chunk_keys = text_k[(text_q == chunk).all(axis=1)]
        assert sorted(map(tuple, chunk_keys)) == [(0, num_video), (num_video, num_tokens)]
    assert plan.text_cu_seqlens_q.tolist() == [0, text_len]
    assert torch.equal(plan.text_page_table, torch.arange(plan.num_pages, dtype=torch.int32)[None])
    assert plan.text_seqused_k.tolist() == [num_tokens]
    return np.bincount(window_block, minlength=num_blocks)


@pytest.mark.parametrize("text_len", [512, 0])
def test_plan_matches_reference_ranges_at_production_size(text_len: int) -> None:
    plan = _plan(_PRODUCTION_GRID, text_len)
    window_rows_per_block = _assert_plan_matches_ranges(plan, _ranges(_PRODUCTION_GRID, text_len))

    assert plan.page_table.dtype == plan.seqused_k.dtype == plan.cu_seqlens_q.dtype == torch.int32
    assert plan.page_table.shape == (4080, 1000 + text_len // 16)
    assert (window_rows_per_block.min(), window_rows_per_block.max(), window_rows_per_block.sum()) == (27, 125, 386784)
    assert window_rows_per_block.mean() == pytest.approx(94.8, abs=0.05)


def test_plan_production_key_statistics() -> None:
    plan = _plan(_PRODUCTION_GRID, 512)
    video_keys = plan.seqused_k.long() - 512

    assert (int(video_keys.min()), int(video_keys.max()), int(video_keys.sum())) == (3312, 16000, 48_962_304)
    assert float(video_keys.double().mean()) == pytest.approx(12000.6, abs=0.05)
    assert int(video_keys.max()) // plan.page_size == 1000
    assert (plan.max_seqlen_q, plan.max_seqlen_k) == (128, 16000 + 512)
    assert (plan.num_pages, plan.num_padded_tokens) == (32162, 514_592)
    assert plan.cu_seqlens_q.shape == (4081,)


def _block_major_coords(grid: tuple[int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Independent block-major order (raster inside a block) and each position's block coordinates."""

    t, h, w = (axis.ravel() for axis in np.meshgrid(*(np.arange(n) for n in grid), indexing="ij"))
    bt, bh, bw = _WINDOW.block_size
    block = np.stack((t // bt, h // bh, w // bw), axis=1)
    order = np.lexsort((w, h, t, block[:, 2], block[:, 1], block[:, 0]))
    return order, block[order]


def _closed_form_mask(block: np.ndarray, query: np.ndarray, text_len: int) -> np.ndarray:
    """Keys of the block-major positions ``query``: the in-radius video blocks and every text token."""

    num_video = block.shape[0]
    coords, block_id = np.unique(block, axis=0, return_inverse=True)
    near = (np.abs(coords[:, None, :] - coords[None, :, :]) <= np.array(_WINDOW.radius)).all(axis=-1)
    mask = np.ones((query.shape[0], num_video + text_len), dtype=bool)
    is_video = query < num_video
    mask[is_video, :num_video] = near[block_id[query[is_video]]][:, block_id]
    return mask


def _plan_key_tokens(plan: Magi2WindowAttentionPlan, pages: torch.Tensor, used: int) -> torch.Tensor:
    pages = pages[: -(-used // plan.page_size)].long()
    tokens = (pages[:, None] * plan.page_size + torch.arange(plan.page_size)).flatten()[:used]
    assert tokens.unique().numel() == used
    return tokens


@pytest.mark.parametrize(("grid", "text_len"), [((27, 8, 24), 37), ((23, 16, 40), 5), ((23, 16, 40), 0)])
def test_window_masks_agree_on_small_grids(grid: tuple[int, int, int], text_len: int) -> None:
    ranges = build_reference_window_ranges(grid, text_len)
    plan = build_window_attention_plan(grid, text_len)
    num_video = ranges.num_video_tokens
    num_tokens = num_video + text_len
    order, block = _block_major_coords(grid)
    assert np.array_equal(ranges.token_order[:num_video].numpy(), order)

    # Query groups: the video blocks, then the text queries in dense-row chunks.
    chunk = _WINDOW.dense_query_chunk
    groups = ranges.block_ranges.long().tolist() + [
        [s, min(s + chunk, num_tokens)] for s in range(num_video, num_tokens, chunk)
    ]
    group_of = {tuple(g): i for i, g in enumerate(groups)}

    closed = _closed_form_mask(block, np.array([s for s, _ in groups]), text_len)
    for start, end in groups[: plan.num_blocks]:
        assert (block[start:end] == block[start]).all(), "a query block mixes latent blocks"

    coverage = np.zeros((len(groups), num_tokens), dtype=np.int16)
    for (qs, qe), (ks, ke) in zip(ranges.q_ranges.tolist(), ranges.k_ranges.tolist()):
        coverage[group_of[qs, qe], ks:ke] += 1
    assert coverage.max() == 1, "reference ranges cover a (query, key) pair twice"

    from_plan = np.zeros((len(groups), num_tokens), dtype=bool)
    for b in range(plan.num_blocks):
        from_plan[b, _plan_key_tokens(plan, plan.page_table[b], int(plan.seqused_k[b])).numpy()] = True
    if text_len:
        text_keys = _plan_key_tokens(plan, plan.text_page_table[0], int(plan.text_seqused_k[0])).numpy()
        from_plan[plan.num_blocks :, text_keys] = True

    assert np.array_equal(coverage == 1, closed)
    assert np.array_equal(from_plan, closed)
    # The window is local: some video pairs are masked on every grid used here.
    assert not closed[: plan.num_blocks, :num_video].all()


def _masked_sdpa(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: np.ndarray, scale: float | None):
    group = q.shape[1] // k.shape[1]
    k = k.repeat_interleave(group, dim=1).transpose(0, 1)[None]
    v = v.repeat_interleave(group, dim=1).transpose(0, 1)[None]
    out = []
    for start in range(0, q.shape[0], 512):
        q_chunk = q[start : start + 512].transpose(0, 1)[None]
        attn_mask = torch.from_numpy(mask[start : start + 512])
        out.append(F.scaled_dot_product_attention(q_chunk, k, v, attn_mask=attn_mask, scale=scale)[0].transpose(0, 1))
    return torch.cat(out)


@pytest.mark.parametrize(("num_heads_q", "num_heads_kv", "softmax_scale"), [(4, 1, None), (8, 2, None), (8, 2, 0.1)])
def test_torch_reference_matches_masked_sdpa(num_heads_q: int, num_heads_kv: int, softmax_scale: float | None) -> None:
    grid, text_len, head_dim = (27, 8, 24), 37, 32
    plan = build_window_attention_plan(grid, text_len)
    num_tokens = plan.num_tokens
    assert num_tokens % plan.page_size != 0  # the last text page is partial
    gen = torch.Generator().manual_seed(num_heads_q * 10 + num_heads_kv)
    q = torch.randn(num_tokens, num_heads_q, head_dim, generator=gen)
    k = torch.randn(num_tokens, num_heads_kv, head_dim, generator=gen)
    v = torch.randn(num_tokens, num_heads_kv, head_dim, generator=gen)
    _, block = _block_major_coords(grid)
    mask = _closed_form_mask(block, np.arange(num_tokens), text_len)

    out = torch_window_attention_reference(q, k, v, plan, softmax_scale=softmax_scale)

    torch.testing.assert_close(out, _masked_sdpa(q, k, v, mask, softmax_scale), rtol=1e-5, atol=1e-5)
    out_bf16 = torch_window_attention_reference(q.bfloat16(), k.bfloat16(), v.bfloat16(), plan)
    assert out_bf16.dtype == torch.bfloat16


def test_torch_reference_refuses_large_or_mismatched_inputs() -> None:
    plan = _plan(_PRODUCTION_GRID, 512)
    q = torch.empty(plan.num_tokens, 4, 128, device="meta")
    kv = torch.empty(plan.num_tokens, 1, 128, device="meta")
    with pytest.raises(ValueError, match="limited to 65536 tokens"):
        torch_window_attention_reference(q, kv, kv, plan)

    small = build_window_attention_plan((9, 4, 8), 3)
    q = torch.zeros(small.num_tokens, 2, 8)
    kv = torch.zeros(small.num_tokens, 1, 8)
    with pytest.raises(ValueError, match="limited to 100 tokens"):
        torch_window_attention_reference(q, kv, kv, small, max_tokens=100)
    with pytest.raises(ValueError, match="do not match the plan"):
        torch_window_attention_reference(q[1:], kv[1:], kv[1:], small)
    with pytest.raises(ValueError, match="multiple of key/value heads"):
        torch_window_attention_reference(
            torch.zeros(small.num_tokens, 3, 8),
            torch.zeros(small.num_tokens, 2, 8),
            torch.zeros(small.num_tokens, 2, 8),
            small,
        )
    with pytest.raises(ValueError, match="key/value"):
        torch_window_attention_reference(q, kv, kv[:, :, :4], small)


@pytest.mark.parametrize(
    ("grid", "page_size"), [((27, 6, 24), 16), ((27, 8, 22), 16), ((27, 8, 24), 32), ((27, 8, 24), 0)]
)
def test_plan_refuses_grids_without_page_aligned_blocks(grid: tuple[int, int, int], page_size: int) -> None:
    with pytest.raises(ValueError, match="paged window attention"):
        build_window_attention_plan(grid, 7, page_size=page_size)


@pytest.mark.parametrize("page_size", [1, 2, 4, 8])
def test_plan_keeps_key_sets_for_every_accepted_page_size(page_size: int) -> None:
    # (27, 8, 24) ends with a partial 3-frame temporal block.
    grid, text_len = (27, 8, 24), 37
    plan = build_window_attention_plan(grid, text_len, page_size=page_size)
    reference = _plan(grid, text_len)

    assert plan.page_size == page_size
    assert torch.equal(plan.cu_seqlens_q, reference.cu_seqlens_q)
    assert not (plan.cu_seqlens_q % page_size).any()
    assert torch.equal(plan.seqused_k, reference.seqused_k)
    for b in range(plan.num_blocks):
        keys = _plan_key_tokens(plan, plan.page_table[b], int(plan.seqused_k[b]))
        reference_keys = _plan_key_tokens(reference, reference.page_table[b], int(reference.seqused_k[b]))
        assert torch.equal(keys.sort().values, reference_keys.sort().values)


def test_plan_pads_pages_with_zeros_and_moves_devices() -> None:
    plan = build_window_attention_plan((9, 4, 8), 21)
    assert (plan.num_tokens, plan.num_pages, plan.num_padded_tokens) == (309, 20, 320)
    x = torch.randn(plan.num_tokens, 2, 4)

    pages = plan.pad_to_pages(x)

    assert pages.shape == (20, 16, 2, 4)
    assert torch.equal(pages.flatten(0, 1)[: plan.num_tokens], x)
    assert not pages.flatten(0, 1)[plan.num_tokens :].any()
    with pytest.raises(ValueError, match="expected 309 tokens"):
        plan.pad_to_pages(x[1:])

    moved = plan.to("meta")
    assert moved.page_table.device.type == "meta"
    assert moved.text_page_table.device.type == "meta"
    assert moved.text_split_page_table.device.type == moved.text_split_seqused_k.device.type == "meta"
    assert plan.page_table.device.type == "cpu"
    assert (moved.grid, moved.text_len, moved.max_seqlen_k) == (plan.grid, plan.text_len, plan.max_seqlen_k)


def test_plan_without_text_has_no_text_pages() -> None:
    plan = build_window_attention_plan((9, 4, 8), 0)

    assert plan.text_cu_seqlens_q.tolist() == [0, 0]
    assert plan.text_seqused_k.tolist() == [plan.num_tokens]
    assert torch.equal(plan.seqused_k % plan.page_size, torch.zeros_like(plan.seqused_k))
    out = torch_window_attention_reference(
        torch.randn(plan.num_tokens, 2, 8), torch.randn(plan.num_tokens, 1, 8), torch.randn(plan.num_tokens, 1, 8), plan
    )
    assert torch.isfinite(out).all()


def _text_split_keys(plan: Magi2WindowAttentionPlan) -> list[np.ndarray]:
    """Every text query's key tokens, concatenated over the paged text rows that hold it."""

    cu = plan.text_split_cu_seqlens_q.long().numpy()
    table = plan.text_split_page_table.long().numpy()
    used = plan.text_split_seqused_k.long().numpy()
    keys: list[list[np.ndarray]] = [[] for _ in range(plan.text_len)]
    for row in range(table.shape[0]):
        pages = table[row, : -(-used[row] // plan.page_size)]
        tokens = (pages[:, None] * plan.page_size + np.arange(plan.page_size)).ravel()[: used[row]]
        for position in range(cu[row], cu[row + 1]):
            keys[position % plan.text_len].append(tokens)
    return [np.concatenate(query_keys) for query_keys in keys]


@pytest.mark.parametrize(
    ("grid", "text_len", "key_splits"),
    [
        ((1, 4, 4), 1, 32),  # more splits than pages
        ((9, 4, 8), 1, 32),
        ((9, 4, 8), 37, 4),  # a shorter last run of pages
        ((1, 4, 4), 120, 4),  # 9 pages: 3 runs of 3, not 4 runs
        ((9, 4, 8), 128, 32),
        ((9, 4, 8), 130, 7),  # two query chunks per run
        ((11, 8, 12), 300, 32),  # three query chunks per run
        ((11, 8, 12), 300, 1),
    ],
)
def test_text_split_rows_attend_every_key_once_per_query(
    grid: tuple[int, int, int], text_len: int, key_splits: int
) -> None:
    plan = build_window_attention_plan(grid, text_len, text_key_splits=key_splits)
    cu = plan.text_split_cu_seqlens_q.long()
    num_rows = plan.text_split_page_table.shape[0]
    splits = plan.text_key_splits

    for tensor in (plan.text_split_cu_seqlens_q, plan.text_split_page_table, plan.text_split_seqused_k):
        assert tensor.dtype == torch.int32 and tensor.is_contiguous()
    assert plan.query_chunk == 128
    assert 1 <= splits <= min(key_splits, plan.num_pages)
    assert num_rows == splits * -(-text_len // plan.query_chunk) == cu.numel() - 1
    assert plan.text_split_seqused_k.shape == (num_rows,)
    assert (int(cu[0]), int(cu[-1])) == (0, splits * text_len)
    assert 0 < int(cu.diff().min()) and int(cu.diff().max()) <= plan.query_chunk
    # A row holds queries of one copy only, so the partial results reshape to [splits, text_len].
    assert torch.equal(cu[:-1] // text_len, (cu[1:] - 1) // text_len)
    assert int(plan.text_split_seqused_k.min()) > 0
    assert int(plan.text_split_page_table.max()) < plan.num_pages
    assert plan.text_split_max_seqlen_k == int(plan.text_split_seqused_k.max())
    for keys in _text_split_keys(plan):
        assert np.array_equal(np.sort(keys), np.arange(plan.num_tokens))


def test_text_split_rows_at_production_size() -> None:
    plan = _plan(_PRODUCTION_GRID, 512)

    assert (plan.query_chunk, plan.text_key_splits) == (128, 32)
    assert plan.text_split_page_table.shape == (4 * 32, 1006)
    assert plan.text_split_cu_seqlens_q[:6].tolist() == [0, 128, 256, 384, 512, 640]
    assert int(plan.text_split_cu_seqlens_q[-1]) == 32 * 512
    assert plan.text_split_seqused_k[:4].tolist() == 4 * [1006 * 16]
    assert plan.text_split_seqused_k[-4:].tolist() == 4 * [514_592 - 31 * 1006 * 16]
    assert plan.text_split_max_seqlen_k == 1006 * 16

    without_text = _plan(_PRODUCTION_GRID, 0)
    assert without_text.text_key_splits == 0
    assert without_text.text_split_cu_seqlens_q.tolist() == [0]
    assert without_text.text_split_page_table.shape == (0, 0)
    assert without_text.text_split_seqused_k.numel() == 0 and without_text.text_split_max_seqlen_k == 0


def test_plan_refuses_non_positive_text_key_splits() -> None:
    with pytest.raises(ValueError, match="text_key_splits"):
        build_window_attention_plan((9, 4, 8), 3, text_key_splits=0)


@pytest.mark.parametrize("builder", [build_reference_window_ranges, build_window_attention_plan])
def test_builders_reject_bad_grids(builder) -> None:
    with pytest.raises(ValueError, match="three positive"):
        builder((0, 4, 8), 3)
    with pytest.raises(ValueError, match="three positive"):
        builder((4, 8), 3)
    with pytest.raises(ValueError, match="non-negative"):
        builder((8, 4, 8), -1)


def _qkv(plan: Magi2WindowAttentionPlan, num_heads_q: int = 4, num_heads_kv: int = 2, device: str = "cpu"):
    generator = torch.Generator().manual_seed(7)
    shape = (plan.num_tokens, num_heads_q, 16)
    q = torch.randn(shape, generator=generator).to(device=device, dtype=torch.bfloat16)
    k = torch.randn(plan.num_tokens, num_heads_kv, 16, generator=generator).to(device=device, dtype=torch.bfloat16)
    v = torch.randn(plan.num_tokens, num_heads_kv, 16, generator=generator).to(device=device, dtype=torch.bfloat16)
    return q, k, v


def test_window_attention_kernel_runs_the_plan_through_the_shared_attention_layer() -> None:
    plan = _plan((9, 4, 8), 21)
    q, k, v = _qkv(plan)
    layer = Attention(
        num_heads=4,
        num_kv_heads=2,
        head_size=16,
        causal=False,
        softmax_scale=0.3,
        qkv_layout="THD",
        skip_sequence_parallel=True,
        disable_kv_quant=True,
        custom_attention=Magi2WindowAttentionKernel(0.3),
    )
    metadata = AttentionMetadata(extra={"magi2_window_plan": plan, "magi2_split_sizes": [plan.num_tokens]})

    out = layer(q, k, v, metadata)

    assert out.dtype == torch.bfloat16
    assert torch.equal(out, torch_window_attention_reference(q, k, v, plan, softmax_scale=0.3))
    assert torch.equal(window_attention(q, k, v, plan, softmax_scale=0.3), out)


def test_window_attention_kernel_requires_plan_and_split_metadata() -> None:
    plan = _plan((9, 4, 8), 3)
    q, k, v = _qkv(plan)
    kernel = Magi2WindowAttentionKernel(0.25)

    with pytest.raises(ValueError, match="requires attention metadata"):
        kernel(q, k, v, None)
    with pytest.raises(TypeError, match="magi2_window_plan"):
        kernel(q, k, v, AttentionMetadata(extra={"magi2_split_sizes": [plan.num_tokens]}))
    with pytest.raises(TypeError, match="magi2_split_sizes"):
        kernel(q, k, v, AttentionMetadata(extra={"magi2_window_plan": plan}))


def test_ulysses_window_attention_checks_the_split_sizes() -> None:
    plan = _plan((9, 4, 8), 3)
    q, k, v = _qkv(plan)
    single = Magi2ParallelGroup(None, 1, 0)
    pair = Magi2ParallelGroup(None, 2, 1)
    half = [plan.num_tokens - plan.num_tokens // 2, plan.num_tokens // 2]

    with pytest.raises(ValueError, match="do not partition"):
        ulysses_window_attention(q, k, v, plan, [plan.num_tokens - 1], group=single)
    with pytest.raises(ValueError, match="do not partition"):
        ulysses_window_attention(q, k, v, plan, [plan.num_tokens], group=pair)
    with pytest.raises(ValueError, match="owns"):
        ulysses_window_attention(q, k, v, plan, half, group=pair)


def test_window_attention_refuses_device_tensors_without_a_paged_kernel() -> None:
    plan = _plan((9, 4, 8), 3)
    q, k, v = _qkv(plan, device="meta")

    with pytest.raises(NotImplementedError, match="no paged kernel for meta tensors"):
        window_attention(q, k, v, plan)
