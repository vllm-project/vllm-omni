# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of the packed Flow attention metadata for FA3, FA2 and FlashInfer.

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


def _attention(lengths, backend, chunk_size=None, width=None):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import (
        build_row_attention,
        pack_rows,
    )

    rows = pack_rows(lengths, torch.device("cpu"), width=width)
    attention = build_row_attention(rows, heads=2, head_dim=8, backend=backend, chunk_size=chunk_size)
    return rows, attention


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_attention_metadata_retains_live_graph_buffers(backend):
    from dataclasses import replace

    _, source = _attention([17, 0, 33], backend, chunk_size=CHUNK)
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
    assert graph_attention.page_size == (16 if backend == "fa2" else 1)


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


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("chunk_size", [None, CHUNK])
def test_paged_attention_matches_isolated_row_attention(backend, chunk_size):
    # Unaligned rows, a 16-frame row, a single frame and an empty bucket slot.
    lengths = [67, 0, 103, 16, 1]
    _, attention = _attention(lengths, backend, chunk_size)
    generator = torch.Generator().manual_seed(0)
    q, k, v = [torch.randn(sum(lengths), 2, 8, generator=generator) for _ in range(3)]

    actual = _paged_attention(q, k, v, attention)

    torch.testing.assert_close(actual, _row_reference(q, k, v, lengths, chunk_size))


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("chunk_size", [None, CHUNK])
@pytest.mark.parametrize("lengths", [[1], [15, 17, 33], [67, 0, 103, 16, 1], [500, 7, 129, 64, 0]])
def test_every_segment_reads_exactly_its_rows_frames(backend, chunk_size, lengths):
    rows, attention = _attention(lengths, backend, chunk_size)
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
    _, attention = _attention(lengths, backend="fa2", chunk_size=chunk_size)
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
    for backend in ("fa2", "fa3"):
        runner = FlowGraphRunner(SimpleNamespace(dit=dit, backend=backend, dtype=None))
        page_size = 16 if backend == "fa2" else 1
        for _ in range(50):
            lengths = tuple(rng.choice([1, 15, 17, rng.randint(1, 900)]) for _ in range(rng.randint(1, 8)))
            streaming = rng.random() < 0.5
            key, twin, seg_rows, seg_ends, _ = runner._layout(lengths, streaming)
            state = runner._allocate(key, noise, spks, time_span)
            if backend == "fa2" and not streaming:
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
    rows, attention = _attention(lengths, backend="fa3", chunk_size=CHUNK)

    assert attention.page_size == 1
    assert torch.equal(attention.slot_frames, torch.arange(rows.total))
    starts = rows.starts_host.long()
    for segment, (row, end) in enumerate([(0, 50), (0, 67), (1, 0), (2, 50), (2, 100), (2, 103)]):
        assert int(attention.cache_seqlens[segment]) == end
        assert torch.equal(attention.page_table[segment, :end].long(), starts[row] + torch.arange(end))
        assert not attention.page_table[segment, end:].any()


def test_fa2_pages_round_up_while_geometry_keeps_the_bucketed_width():
    rows, attention = _attention([5, 0, 37], backend="fa2", chunk_size=CHUNK, width=256)

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
    _, fa2 = _attention(lengths, backend="fa2", width=width)
    _, fa3 = _attention(lengths, backend="fa3", width=width)

    # FA2 full context: one segment per row whose K/V span is its query span.
    assert not fa2.paged and fa2.page_table is None and fa2.slot_frames is None
    assert fa2.cache_seqlens.tolist() == lengths
    assert fa2.cu_seqlens_q.tolist() == [0, 5, 5, 42, 58]
    assert fa2.max_seqlen_q == 37
    # FA3 keeps its one-frame pages, and chunk-causal segments stay paged on both.
    assert fa3.paged
    backends = ("fa2", "fa3", "flashinfer")
    uses_pages = [attention_uses_pages(b, chunked) for b in backends for chunked in (False, True)]
    # FlashInfer plans its own KV indices instead of a FlashAttention page table.
    assert uses_pages == [False, True, True, True, False, False]


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

    supported: set[str] = set()
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime, "_backend_supported", lambda backend: backend in supported)
    monkeypatch.delenv("COSYVOICE3_PACKED_ATTENTION_BACKEND", raising=False)
    return runtime, supported


@pytest.mark.parametrize(
    "supported, pinned, expected",
    [
        ({"fa2", "fa3"}, None, "fa3"),  # Hopper
        ({"fa2"}, None, "fa2"),  # Ampere, Ada, Blackwell
        (set(), None, None),
        ({"fa2", "fa3"}, "fa2", "fa2"),
        ({"fa2"}, "fa3", None),
        # FlashInfer is opt-in: installed alone, it is never picked automatically.
        ({"fa2", "fa3", "flashinfer"}, None, "fa3"),
        ({"flashinfer"}, None, None),
        ({"fa2", "flashinfer"}, "flashinfer", "flashinfer"),
        ({"fa2"}, "flashinfer", None),
    ],
)
def test_packed_backend_prefers_fa3_and_honors_the_pin(monkeypatch, fake_gpu, supported, pinned, expected):
    runtime, available = fake_gpu
    available.update(supported)
    if pinned is not None:
        monkeypatch.setenv("COSYVOICE3_PACKED_ATTENTION_BACKEND", pinned)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    monkeypatch.setenv("COSYVOICE3_PACKED_STREAMING", "1")

    assert runtime.cosyvoice3_packed_attention_backend() == expected
    assert runtime.cosyvoice3_full_response_enabled() is (expected is not None)
    assert runtime.cosyvoice3_packed_streaming_enabled() is (expected is not None)


def test_packed_backend_rejects_unknown_pins(monkeypatch, fake_gpu):
    monkeypatch.setenv("COSYVOICE3_PACKED_ATTENTION_BACKEND", "3")
    with pytest.raises(ValueError, match="must be fa3, fa2 or flashinfer"):
        fake_gpu[0].cosyvoice3_packed_attention_backend()


@pytest.mark.parametrize("capability, expected", [((7, 5), False), ((8, 0), True), ((12, 0), True)])
def test_flashinfer_pin_keeps_the_stock_flow_below_sm80(monkeypatch, capability, expected):
    import vllm_omni.model_executor.models.cosyvoice3.runtime as runtime

    find_spec = runtime.importlib.util.find_spec
    monkeypatch.setattr(
        runtime.importlib.util, "find_spec", lambda name: object() if name == "flashinfer" else find_spec(name)
    )
    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(runtime.torch.cuda, "get_device_capability", lambda: capability)
    monkeypatch.setenv("COSYVOICE3_PACKED_ATTENTION_BACKEND", "flashinfer")
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    runtime._flashinfer_supported.cache_clear()
    try:
        assert runtime.cosyvoice3_packed_attention_backend() == ("flashinfer" if expected else None)
        assert runtime.cosyvoice3_full_response_enabled() is expected
    finally:
        runtime._flashinfer_supported.cache_clear()


def test_packed_profiles_stay_off_without_cuda(monkeypatch):
    import vllm_omni.model_executor.models.cosyvoice3.runtime as runtime

    monkeypatch.setattr(runtime.torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("COSYVOICE3_FULL_RESPONSE_OPTIMIZATIONS", "1")
    assert runtime.cosyvoice3_packed_attention_backend() is None
    assert not runtime.cosyvoice3_full_response_enabled()


@pytest.fixture
def fake_flashinfer(monkeypatch):
    """A flashinfer module whose wrappers record what they are planned with."""
    import sys
    from types import ModuleType

    class Wrapper:
        def __init__(self, float_workspace, layout, use_cuda_graph=False, **buffers):
            self.use_cuda_graph, self.buffers, self.planned = use_cuda_graph, buffers, None

        def reset_workspace_buffer(self, float_workspace, int_workspace):
            self.workspaces = (float_workspace, int_workspace)

        def plan(self, *args, **kwargs):
            self.planned = (args, kwargs)

    module = ModuleType("flashinfer")
    module.BatchPrefillWithPagedKVCacheWrapper = type("Paged", (Wrapper,), {})
    module.BatchPrefillWithRaggedKVCacheWrapper = type("Ragged", (Wrapper,), {})
    monkeypatch.setitem(sys.modules, "flashinfer", module)
    import vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_flashinfer as packed_flashinfer

    monkeypatch.setattr(packed_flashinfer, "_workspaces", {})
    return packed_flashinfer


@pytest.mark.parametrize("lengths", [[1], [15, 17, 33], [67, 0, 103, 16, 1], [500, 7, 129, 64, 0]])
def test_flashinfer_chunk_plan_reads_each_rows_prefix_in_place(fake_flashinfer, lengths):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import row_segments

    plan = fake_flashinfer.FlashInferRowPlan(
        torch.device("cpu"), heads=2, head_dim=8, dtype=torch.bfloat16, chunked=True
    )
    segment_rows, segment_ends, offsets = row_segments(lengths, CHUNK)
    plan.plan(lengths, segment_rows, segment_ends, offsets)
    (qo_indptr, kv_indptr, indices, last_page_len, heads, kv_heads, head_dim, page_size), kwargs = plan.wrapper.planned

    assert (heads, kv_heads, head_dim, page_size) == (2, 2, 8, 1)
    # FlashInfer reads index tensors as int32; int64 ones are misread.
    assert {t.dtype for t in (qo_indptr, kv_indptr, indices, last_page_len)} == {torch.int32}
    assert kwargs == {"causal": False, "q_data_type": torch.bfloat16, "disable_split_kv": True}
    assert qo_indptr.tolist() == offsets
    assert last_page_len.tolist() == [1] * len(segment_ends)
    # One-frame pages are packed frames: each segment reads its row's prefix.
    row_starts = [sum(lengths[:row]) for row in range(len(lengths))]
    for segment, (row, end) in enumerate(zip(segment_rows, segment_ends)):
        frames = indices[kv_indptr[segment] : kv_indptr[segment + 1]]
        assert frames.tolist() == list(range(row_starts[row], row_starts[row] + end))


def test_flashinfer_full_context_plan_attends_each_rows_own_span(fake_flashinfer):
    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import row_segments

    plan = fake_flashinfer.FlashInferRowPlan(
        torch.device("cpu"), heads=2, head_dim=8, dtype=torch.float16, chunked=False
    )
    lengths = [5, 0, 37, 16]
    plan.plan(lengths, *row_segments(lengths, None))
    (qo_indptr, kv_indptr, heads, kv_heads, head_dim), kwargs = plan.wrapper.planned

    assert type(plan.wrapper).__name__ == "Ragged"
    assert qo_indptr.tolist() == kv_indptr.tolist() == [0, 5, 5, 42, 58]
    assert (heads, kv_heads, head_dim) == (2, 2, 8)


@pytest.mark.parametrize("seed", range(4))
def test_flashinfer_graph_buffers_cover_every_padded_layout(monkeypatch, fake_flashinfer, seed):
    import random
    from types import SimpleNamespace

    from vllm_omni.model_executor.models.cosyvoice3.code2wav_core.packed_dit import FlowGraphRunner

    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: None)
    attn = SimpleNamespace(heads=2, inner_dim=16)
    dit = SimpleNamespace(transformer_blocks=[SimpleNamespace(attn=attn)], static_chunk_size=CHUNK)
    runner = FlowGraphRunner(SimpleNamespace(dit=dit, backend="flashinfer", dtype=torch.bfloat16))
    rng = random.Random(seed)
    noise, spks, time_span = torch.zeros(1, 1, 80), torch.zeros(1, 80), torch.zeros(4)
    for _ in range(50):
        lengths = tuple(rng.choice([1, 15, 17, rng.randint(1, 900)]) for _ in range(rng.randint(1, 8)))
        streaming = rng.random() < 0.5
        key, twin, seg_rows, seg_ends, offsets = runner._layout(lengths, streaming)
        state = runner._allocate(key, noise, spks, time_span)
        wrapper = state.flashinfer.wrapper

        assert state.page_table is None and state.slot_frames is None
        assert wrapper.use_cuda_graph
        assert wrapper.buffers["qo_indptr_buf"].shape[0] == len(offsets)
        if streaming:
            assert sum(seg_ends) <= wrapper.buffers["paged_kv_indices_buf"].shape[0]
            assert wrapper.buffers["paged_kv_last_page_len_buf"].shape[0] == len(seg_ends)
        else:
            assert wrapper.buffers["kv_indptr_buf"].shape[0] == len(offsets)
