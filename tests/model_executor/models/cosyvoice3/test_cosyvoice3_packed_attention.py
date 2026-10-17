# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of the packed Flow attention metadata for FA3 and FA2 pages.

The kernels only run on CUDA; here a reference reads the paged KV exactly as
the FlashAttention calls are configured, so the page table, KV slots and query
segments can be checked against chunk-causal attention on unpadded rows.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK = 50


def _attention(lengths, fa_version, chunk_size=None, width=None):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        RaggedRowAttention,
        pack_rows,
    )

    rows = pack_rows(lengths, torch.device("cpu"), width=width)
    attention = RaggedRowAttention(rows, heads=2, head_dim=8, fa_version=fa_version, chunk_size=chunk_size)
    return rows, attention


def _paged_kv(x, attention):
    """Lay packed K or V out as ``packed_fa2`` does (identity for FA3 pages)."""
    page_size = attention.page_size
    pages = x.shape[0] // page_size + attention.page_table.shape[0]
    return x.new_zeros(pages * page_size, *x.shape[1:]).index_copy_(0, attention.kv_slots, x)


def _segment_slots(attention, segment):
    page_size = attention.page_size
    pages = attention.page_table[segment].long()
    slots = (pages[:, None] * page_size + torch.arange(page_size)).flatten()
    return slots[: int(attention.cache_seqlens[segment])]


def _paged_attention(q, k, v, attention):
    """Each query segment attends the KV prefix its page-table row maps."""
    k_pages, v_pages = _paged_kv(k, attention), _paged_kv(v, attention)
    offsets = attention.cu_seqlens_q.tolist()
    output = torch.zeros_like(q)
    for segment in range(attention.cache_seqlens.shape[0]):
        start, end = offsets[segment], offsets[segment + 1]
        if start == end:
            continue
        slots = _segment_slots(attention, segment)
        output[start:end] = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k_pages[slots].transpose(0, 1), v_pages[slots].transpose(0, 1)
        ).transpose(0, 1)
    return output


def _row_reference(q, k, v, lengths, chunk_size):
    output, start = [], 0
    for length in lengths:
        position = torch.arange(length)
        if chunk_size is None:
            mask = torch.ones(length, length, dtype=torch.bool)
        else:
            mask = position[None, :] < (position[:, None] // chunk_size + 1) * chunk_size
        rows = slice(start, start + length)
        output.append(
            F.scaled_dot_product_attention(
                q[rows].transpose(0, 1), k[rows].transpose(0, 1), v[rows].transpose(0, 1), attn_mask=mask
            ).transpose(0, 1)
        )
        start += length
    return torch.cat(output)


@pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("chunk_size", [None, CHUNK])
def test_paged_attention_matches_isolated_row_attention(fa_version, chunk_size):
    # Unaligned rows, a 16-frame row, a single frame and an empty bucket slot.
    lengths = [67, 0, 103, 16, 1]
    _, attention = _attention(lengths, fa_version, chunk_size)
    generator = torch.Generator().manual_seed(0)
    q, k, v = [torch.randn(sum(lengths), 2, 8, generator=generator) for _ in range(3)]

    actual = _paged_attention(q, k, v, attention)

    torch.testing.assert_close(actual, _row_reference(q, k, v, lengths, chunk_size))


@pytest.mark.parametrize("fa_version", [2, 3])
@pytest.mark.parametrize("chunk_size", [None, CHUNK])
@pytest.mark.parametrize("lengths", [[1], [15, 17, 33], [67, 0, 103, 16, 1], [500, 7, 129, 64, 0]])
def test_kv_slots_are_page_aligned_disjoint_and_fit_the_fa2_allocation(fa_version, chunk_size, lengths):
    rows, attention = _attention(lengths, fa_version, chunk_size)
    page_size = attention.page_size
    slots = attention.kv_slots

    assert slots.unique().numel() == rows.total
    row_starts = rows.starts_host[:-1].long()
    nonempty = torch.tensor(lengths) > 0
    assert torch.all(slots[row_starts[nonempty]] % page_size == 0)
    # packed_fa2 sizes its page buffer from shapes alone; every slot must fit.
    pages = rows.total // page_size + attention.page_table.shape[0]
    assert int(slots.max()) < pages * page_size
    # Every segment reads its row's packed frames, in order, up to its end.
    segment_rows = []
    for row, length in enumerate(lengths):
        chunks = 1 if length == 0 else len(range(0, length, chunk_size or length))
        segment_rows += [row] * chunks
    assert len(segment_rows) == attention.cache_seqlens.shape[0]
    for segment, row in enumerate(segment_rows):
        end = int(attention.cache_seqlens[segment])
        expected = slots[row_starts[row] : row_starts[row] + end]
        assert torch.equal(_segment_slots(attention, segment), expected)


def test_fa3_pages_are_the_packed_sequence_itself():
    lengths = [67, 0, 103]
    rows, attention = _attention(lengths, fa_version=3, chunk_size=CHUNK)

    assert attention.page_size == 1
    assert torch.equal(attention.kv_slots, torch.arange(rows.total))
    starts = rows.starts_host.long()
    for segment, (row, end) in enumerate([(0, 50), (0, 67), (1, 0), (2, 50), (2, 100), (2, 103)]):
        assert int(attention.cache_seqlens[segment]) == end
        assert torch.equal(attention.page_table[segment, :end].long(), starts[row] + torch.arange(end))
        assert not attention.page_table[segment, end:].any()


def test_fa2_pages_round_up_while_geometry_keeps_the_bucketed_width():
    rows, attention = _attention([5, 0, 37], fa_version=2, width=256)

    assert rows.width == 256
    assert attention.page_table.shape == (3, 16)
    assert attention.cache_seqlens.tolist() == [5, 0, 37]
    assert attention.cu_seqlens_q.tolist() == [0, 5, 5, 42]
    # Row 2 starts on page 1 although it starts at packed frame 5.
    assert attention.page_table[2, :3].tolist() == [1, 2, 3]
    assert not attention.page_table[1].any()


def test_rope_only_needs_a_width_covering_every_row():
    from types import SimpleNamespace

    from x_transformers.x_transformers import RotaryEmbedding

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import pack_rows, rope_tensor_geometry

    estimator = SimpleNamespace(dit=SimpleNamespace(rotary_embed=RotaryEmbedding(8)))
    exact = rope_tensor_geometry(estimator, pack_rows([67, 103], torch.device("cpu")))
    bucketed = rope_tensor_geometry(estimator, pack_rows([67, 103], torch.device("cpu"), width=256))

    for exact_part, bucketed_part in zip(exact, bucketed):
        torch.testing.assert_close(bucketed_part, exact_part, rtol=0, atol=0)


@pytest.fixture
def fake_gpu(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.runtime as runtime

    supported: set[int] = set()
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "_fa_version_supported", lambda version: version in supported)
    monkeypatch.delenv("COSYVOICE3_PACKED_FA_VERSION", raising=False)
    return runtime, supported


@pytest.mark.parametrize(
    "supported, pinned, expected",
    [
        ({2, 3}, None, 3),  # Hopper
        ({2}, None, 2),  # Ampere, Ada, Blackwell
        (set(), None, None),
        ({2, 3}, "2", 2),
        ({2}, "3", None),
    ],
)
def test_packed_fa_version_prefers_fa3_and_honors_the_pin(monkeypatch, fake_gpu, supported, pinned, expected):
    runtime, available = fake_gpu
    available.update(supported)
    if pinned is not None:
        monkeypatch.setenv("COSYVOICE3_PACKED_FA_VERSION", pinned)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")

    assert runtime.cosyvoice3_packed_fa_version() == expected
    assert runtime.cosyvoice3_full_response_enabled() is (expected is not None)
    assert runtime.cosyvoice3_packed_streaming_enabled() is (expected is not None)


def test_packed_fa_version_rejects_unknown_pins(monkeypatch, fake_gpu):
    monkeypatch.setenv("COSYVOICE3_PACKED_FA_VERSION", "4")
    with pytest.raises(ValueError, match="must be 2 or 3"):
        fake_gpu[0].cosyvoice3_packed_fa_version()


def test_packed_profiles_stay_off_without_cuda(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.runtime as runtime

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    assert runtime.cosyvoice3_packed_fa_version() is None
    assert not runtime.cosyvoice3_full_response_enabled()
