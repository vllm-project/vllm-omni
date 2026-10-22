# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of the packed Flow attention metadata for FA3 and FA2.

The kernels only run on CUDA; here a reference reads the K/V exactly as the
FlashAttention calls are configured (paged, or unpaged varlen for FA2 full
context), so the page table, KV gather index and query segments can be checked
against chunk-causal attention on unpadded rows.
"""

import pytest
import torch
import torch.nn.functional as F

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

CHUNK = 50


def _attention(lengths, fa_version, chunk_size=None, width=None):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        build_row_attention,
        pack_rows,
    )

    rows = pack_rows(lengths, torch.device("cpu"), width=width)
    attention = build_row_attention(rows, heads=2, head_dim=8, fa_version=fa_version, chunk_size=chunk_size)
    return rows, attention


@pytest.mark.parametrize("fa_version", [2, 3])
def test_attention_metadata_retains_live_graph_buffers(fa_version):
    from dataclasses import replace

    _, source = _attention([17, 0, 33], fa_version, chunk_size=CHUNK)
    graph_attention = replace(source)

    # Constructing metadata must retain the graph buffers, not snapshot them.
    for name in ("cache_seqlens", "cu_seqlens_q", "page_table", "slot_frames"):
        buffer = getattr(source, name)
        assert getattr(graph_attention, name) is buffer
        address = buffer.data_ptr()
        updated = torch.zeros_like(buffer)
        buffer.copy_(updated)
        assert getattr(graph_attention, name).data_ptr() == address
        assert torch.equal(getattr(graph_attention, name), updated)
    assert graph_attention.page_size == (16 if fa_version == 2 else 1)


def _segment_frames(attention, segment):
    """Packed frames a segment's K/V reads, the way the kernel reads them."""
    end = int(attention.cache_seqlens[segment])
    if not attention.paged:
        # Unpaged varlen: keys share the query offsets.
        start = int(attention.cu_seqlens_q[segment])
        return torch.arange(start, start + end)
    page_size = attention.page_size
    pages = attention.page_table[segment].long()
    slots = (pages[:, None] * page_size + torch.arange(page_size)).flatten()[:end]
    # Paged K/V is the packed sequence gathered by slot_frames (see paged_kv).
    return attention.slot_frames[slots]


def _paged_attention(q, k, v, attention):
    """Each query segment attends the K/V frames its metadata maps."""
    offsets = attention.cu_seqlens_q.tolist()
    output = torch.zeros_like(q)
    for segment in range(attention.cache_seqlens.shape[0]):
        start, end = offsets[segment], offsets[segment + 1]
        if start == end:
            continue
        frames = _segment_frames(attention, segment)
        output[start:end] = F.scaled_dot_product_attention(
            q[start:end].transpose(0, 1), k[frames].transpose(0, 1), v[frames].transpose(0, 1)
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
def test_every_segment_reads_exactly_its_rows_frames(fa_version, chunk_size, lengths):
    rows, attention = _attention(lengths, fa_version, chunk_size)
    page_size = attention.page_size

    if attention.paged:
        # Rows start on a page and waste less than one page each.
        slot_frames = attention.slot_frames
        assert slot_frames.shape[0] == sum(-(-n // page_size) for n in lengths) * page_size
        assert int(attention.page_table.max()) < slot_frames.shape[0] // page_size
    # Paged slots past a row's end alias frame 0, so a segment must never read
    # past its own row: each one reads exactly its row's frames, in order.
    row_starts = rows.starts_host[:-1].long()
    segment_rows = []
    for row, length in enumerate(lengths):
        chunks = 1 if length == 0 else len(range(0, length, chunk_size or length))
        segment_rows += [row] * chunks
    assert len(segment_rows) == attention.cache_seqlens.shape[0]
    for segment, row in enumerate(segment_rows):
        end = int(attention.cache_seqlens[segment])
        assert torch.equal(_segment_frames(attention, segment), row_starts[row] + torch.arange(end))


@pytest.mark.parametrize("chunk_size", [None, CHUNK])
def test_fa2_slots_aliasing_frame_zero_never_reach_other_rows(chunk_size):
    lengths = [7, 67, 1, 103, 16]
    _, attention = _attention(lengths, fa_version=2, chunk_size=chunk_size)
    generator = torch.Generator().manual_seed(1)
    q, k, v = [torch.randn(sum(lengths), 2, 8, generator=generator) for _ in range(3)]
    clean = _paged_attention(q, k, v, attention)
    k[0], v[0] = float("nan"), float("nan")

    poisoned = _paged_attention(q, k, v, attention)

    assert torch.equal(poisoned[lengths[0] :], clean[lengths[0] :])


@pytest.mark.parametrize("seed", range(4))
def test_flow_graph_slot_buffer_covers_every_padded_layout(monkeypatch, seed):
    import random
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        FlowGraphRunner,
        pack_rows,
        page_metadata,
    )

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: None)
    attn = SimpleNamespace(heads=2, inner_dim=16)
    dit = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=attn)], static_chunk_size=CHUNK)
    rng = random.Random(seed)
    noise, spks, time_span = torch.zeros(1, 1, 80), torch.zeros(1, 80), torch.zeros(4)
    for fa_version in (2, 3):
        runner = FlowGraphRunner(SimpleNamespace(dit=dit, fa_version=fa_version))
        page_size = 16 if fa_version == 2 else 1
        for _ in range(50):
            lengths = tuple(rng.choice([1, 15, 17, rng.randint(1, 900)]) for _ in range(rng.randint(1, 8)))
            streaming = rng.random() < 0.5
            key, twin, seg_rows, seg_ends, _ = runner._layout(lengths, streaming)
            state = runner._allocate(key, noise, spks, time_span)
            if fa_version == 2 and not streaming:
                # Full-context FA2 graphs attend unpaged and keep no page buffers.
                assert state.page_table is None and state.slot_frames is None
                continue
            twin_rows = pack_rows(twin, torch.device("cpu"), width=state.frames.shape[0])
            page_table, slot_frames = page_metadata(
                torch.tensor(twin),
                torch.tensor(seg_rows),
                torch.tensor(seg_ends),
                twin_rows.row_ids,
                twin_rows.positions,
                page_size=page_size,
                columns=state.page_table.shape[1],
                slots=state.slot_frames.shape[0],
            )
            assert sum(-(-n // page_size) for n in twin) * page_size <= state.slot_frames.shape[0]
            assert page_table.shape == state.page_table.shape
            assert int(page_table.max()) < state.slot_frames.shape[0] // page_size


def test_fa3_pages_are_the_packed_sequence_itself():
    lengths = [67, 0, 103]
    rows, attention = _attention(lengths, fa_version=3, chunk_size=CHUNK)

    assert attention.page_size == 1
    assert torch.equal(attention.slot_frames, torch.arange(rows.total))
    starts = rows.starts_host.long()
    for segment, (row, end) in enumerate([(0, 50), (0, 67), (1, 0), (2, 50), (2, 100), (2, 103)]):
        assert int(attention.cache_seqlens[segment]) == end
        assert torch.equal(attention.page_table[segment, :end].long(), starts[row] + torch.arange(end))
        assert not attention.page_table[segment, end:].any()


def test_fa2_pages_round_up_while_geometry_keeps_the_bucketed_width():
    rows, attention = _attention([5, 0, 37], fa_version=2, chunk_size=CHUNK, width=256)

    assert rows.width == 256
    assert attention.page_table.shape == (3, 16)
    assert attention.cache_seqlens.tolist() == [5, 0, 37]
    assert attention.cu_seqlens_q.tolist() == [0, 5, 5, 42]
    # Row 2 starts on page 1 although it starts at packed frame 5.
    assert attention.page_table[2, :3].tolist() == [1, 2, 3]
    assert not attention.page_table[1].any()


@pytest.mark.parametrize("width", [None, 256])
def test_fa2_full_context_attends_unpaged_while_fa3_keeps_in_place_pages(width):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import attention_uses_pages

    lengths = [5, 0, 37, 16]
    _, fa2 = _attention(lengths, fa_version=2, width=width)
    _, fa3 = _attention(lengths, fa_version=3, width=width)

    # FA2 full context: one segment per row whose K/V span is its query span.
    assert not fa2.paged and fa2.page_table is None and fa2.slot_frames is None
    assert fa2.cache_seqlens.tolist() == lengths
    assert fa2.cu_seqlens_q.tolist() == [0, 5, 5, 42, 58]
    assert fa2.max_seqlen_q == 37
    # FA3 keeps its one-frame pages, and chunk-causal segments stay paged on both.
    assert fa3.paged
    assert [attention_uses_pages(v, chunked) for v in (2, 3) for chunked in (False, True)] == [False, True, True, True]


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
