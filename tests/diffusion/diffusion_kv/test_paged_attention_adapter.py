# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from torch import nn
from vllm.v1.kv_cache_interface import FullAttentionSpec

from vllm_omni.diffusion.attention.backends.flash_attn import FlashAttentionImpl
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.diffusion_kv import paged_attention_adapter as adapter_module
from vllm_omni.diffusion.diffusion_kv.paged_attention_adapter import (
    DiffusionPagedAttentionAdapter,
    DiffusionPagedAttentionMetadata,
    DiffusionPagedAttentionRow,
    DiffusionPagedAttentionRowBinding,
    DiffusionPagedAttentionRuntime,
)
from vllm_omni.diffusion.forward_context import (
    get_forward_context,
    override_paged_kv_adapter,
    set_forward_context,
    set_forward_context_denoise_step_idx,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _FakeBlockTables:
    def __init__(self) -> None:
        self.max_num_reqs = 4
        self.max_num_batched_tokens = 16
        self.cp_size = 1
        self.cp_rank = 0
        self.cp_interleave = 1
        self.block_sizes = [4]
        self.blocks_per_kv_block = [1]
        self.num_blocks = SimpleNamespace(np=np.full((1, self.max_num_reqs), 4, dtype=np.int32))
        self.gather_calls: list[tuple[torch.Tensor, int]] = []
        self.slot_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]] = []

    def gather_block_tables(
        self,
        idx_mapping: torch.Tensor,
        num_reqs_padded: int,
    ) -> tuple[torch.Tensor, ...]:
        self.gather_calls.append((idx_mapping.clone(), num_reqs_padded))
        return (torch.tensor([[3, 4], [7, 8]], dtype=torch.int32),)

    def compute_slot_mappings(
        self,
        idx_mapping: torch.Tensor,
        query_start_loc: torch.Tensor,
        positions: torch.Tensor,
        num_tokens_padded: int,
    ) -> torch.Tensor:
        self.slot_calls.append(
            (
                idx_mapping.clone(),
                query_start_loc.clone(),
                positions.clone(),
                num_tokens_padded,
            )
        )
        return positions[:num_tokens_padded].to(torch.int64).unsqueeze(0)


class _FakeSpec:
    non_causal: bool
    block_size = 4

    def __init__(self, *, non_causal: bool) -> None:
        self.non_causal = non_causal

    def max_num_blocks_per_req(self, _vllm_config, max_len: int) -> int:
        return math.ceil(max_len / self.block_size)


class _FakeAttentionGroup:
    def __init__(self, reorder_batch_threshold: int | None) -> None:
        self.builder = SimpleNamespace(reorder_batch_threshold=reorder_batch_threshold)

    def get_metadata_builder(self, _index: int):
        return self.builder


@dataclass(frozen=True, slots=True)
class _FakeNativeMetadata:
    build_id: int
    causal: bool
    seq_lens: torch.Tensor
    query_start_loc_cpu: torch.Tensor
    positions: torch.Tensor
    slot_mappings: torch.Tensor


class _FakeLayer:
    def __init__(self, *, non_causal: bool, num_heads: int = 2, num_kv_heads: int = 2) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = 4
        self.head_size_v = 4
        self.spec = _FakeSpec(non_causal=non_causal)
        self.updates: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, object]] = []
        self.native_events: list[str] = []
        self.layer_name = "layer-0"
        self.kv_cache = object()
        self.attn_backend = SimpleNamespace(forward_includes_kv_cache_update=False)
        self.impl = _FakeNativeImpl(self)


class _FakeNativeImpl:
    def __init__(self, layer: _FakeLayer) -> None:
        self.layer = layer

    def do_kv_cache_update(
        self,
        _layer,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        _kv_cache,
        slot_mapping: torch.Tensor,
    ) -> None:
        self.layer.native_events.append("update")
        self.layer.updates.append((key, value, slot_mapping))

    def forward(
        self,
        _layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        _kv_cache,
        metadata: object,
        output: torch.Tensor,
    ) -> torch.Tensor:
        self.layer.native_events.append("forward")
        self.layer.calls.append((query, key, value, metadata))
        return output.copy_(query.reshape_as(output))


def _make_adapter(
    monkeypatch: pytest.MonkeyPatch,
    *,
    non_causal: bool = True,
    reorder_batch_threshold: int | None = None,
    num_heads: int = 2,
    num_kv_heads: int = 2,
    capacity: int = 16,
) -> tuple[DiffusionPagedAttentionAdapter, _FakeBlockTables, _FakeLayer, list[tuple]]:
    events: list[tuple] = []
    monkeypatch.setattr(
        adapter_module.current_omni_platform,
        "requires_diffusion_paged_kv_prewrite",
        lambda: False,
    )
    block_tables = _FakeBlockTables()
    block_tables.max_num_batched_tokens = capacity
    block_tables.num_blocks.np.fill(math.ceil(capacity / block_tables.block_sizes[0]))
    layer = _FakeLayer(non_causal=non_causal, num_heads=num_heads, num_kv_heads=num_kv_heads)
    kv_cache_config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["layer-0"], kv_cache_spec=layer.spec)],
    )
    row_map = {
        ("req-0", 0, None): DiffusionPagedAttentionRowBinding(2, capacity),
        ("req-1", 1, None): DiffusionPagedAttentionRowBinding(3, capacity),
        ("req-0", None, "text"): DiffusionPagedAttentionRowBinding(1, capacity),
    }

    monkeypatch.setattr(adapter_module, "set_current_vllm_config", lambda _config: nullcontext())

    def build_metadata(**kwargs):
        metadata = _FakeNativeMetadata(
            build_id=len(events),
            causal=kwargs["causal"],
            seq_lens=kwargs["seq_lens"].clone(),
            query_start_loc_cpu=kwargs["query_start_loc_cpu"].clone(),
            positions=kwargs["positions"].clone(),
            slot_mappings=kwargs["slot_mappings"].clone(),
        )
        events.append(("build", kwargs, metadata))
        return {"layer-0": metadata}

    monkeypatch.setattr(adapter_module.current_omni_platform, "build_diffusion_kv_attn_metadata", build_metadata)
    monkeypatch.setattr(
        adapter_module,
        "build_slot_mappings_by_layer",
        lambda slot_mappings, _config: {"layer-0": slot_mappings[0]},
    )

    config = SimpleNamespace(
        name="vllm-config",
        model_config=SimpleNamespace(dtype=torch.float32),
    )
    adapter = DiffusionPagedAttentionAdapter(
        vllm_config=config,
        device=torch.device("cpu"),
        kv_cache_config=kv_cache_config,
        block_tables=block_tables,
        attn_groups=[[_FakeAttentionGroup(reorder_batch_threshold)]],
        layers={"layer-0": layer},
        resolve_row=lambda request_id, sequence_id, context_id: row_map[(request_id, sequence_id, context_id)],
    )
    return adapter, block_tables, layer, events


def _run_omni_paged_backend(context):
    backend = FlashAttentionImpl(
        num_heads=context.layer.num_heads,
        head_size=context.layer.head_size,
        softmax_scale=0.5,
        num_kv_heads=context.layer.num_kv_heads,
    )
    return backend.forward_paged(context)


def test_prepare_batch_reuses_native_block_table_and_metadata_builders(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, block_tables, _, events = _make_adapter(monkeypatch)
    rows = (
        DiffusionPagedAttentionRow(
            request_id="req-0",
            sequence_id=0,
            query_len=3,
            seq_len=8,
            kv_start_pos=4,
        ),
        DiffusionPagedAttentionRow(
            request_id="req-1",
            sequence_id=1,
            query_len=2,
            seq_len=2,
        ),
    )

    batch = adapter.prepare_batch(rows)

    assert batch.num_tokens == 5
    assert batch.row_indices.tolist() == [2, 3]
    assert batch.query_start_loc.tolist() == [0, 3, 5]
    assert batch.seq_lens.tolist() == [8, 2]
    assert batch.positions.tolist() == [4, 5, 6, 0, 1]
    assert block_tables.gather_calls[0][1] == 2
    assert block_tables.gather_calls[0][0].tolist() == [2, 3]
    assert block_tables.slot_calls[0][3] == 5
    build_kwargs = events[0][1]
    assert build_kwargs["causal"] is False
    assert build_kwargs["max_query_len"] == 3
    assert build_kwargs["max_seq_len"] == 8
    assert build_kwargs["positions"] is batch.positions
    assert batch.attn_metadata["layer-0"] is events[0][2]
    assert events[0][2].build_id == 0
    assert batch.slot_mappings_by_layer["layer-0"].tolist() == [4, 5, 6, 0, 1]


def test_omni_paged_backend_consumes_context_and_restores_diffusion_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, layer, events = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=5,
                seq_len=5,
            )
        ]
    )
    query = torch.randn(1, 5, 2, 4)
    key = torch.randn_like(query)
    value = torch.randn_like(query)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, key, value)
        output = _run_omni_paged_backend(context)

    assert output.shape == query.shape
    assert torch.equal(output, query)
    assert layer.calls[0][0].shape == (5, 2, 4)
    assert layer.native_events == ["update", "forward"]
    assert layer.calls[0][3] is events[0][2]


def test_paged_adapter_accepts_native_gqa_shapes(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch, num_heads=32, num_kv_heads=8)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    query = torch.randn(1, 3, 32, 4)
    key = torch.randn(1, 3, 8, 4)
    value = torch.randn_like(key)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, key, value)

    assert layer.num_heads == 32
    assert layer.num_kv_heads == 8
    assert context.query.shape == (3, 32, 4)
    assert context.key_write.shape == (3, 8, 4)
    assert context.value_write.shape == (3, 8, 4)


def test_paged_adapter_preserves_strided_kv_for_platform_writer(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=5, seq_len=5)]
    )
    packed_qkv = torch.randn(1, 5, 3 * 2 * 4)
    query, key, value = packed_qkv.split(2 * 4, dim=-1)
    query = query.reshape(1, 5, 2, 4)
    key = key.reshape(1, 5, 2, 4)
    value = value.reshape(1, 5, 2, 4)
    assert not key.is_contiguous()
    assert not value.is_contiguous()

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, key, value)

    assert context.query.is_contiguous()
    assert not context.key_write.is_contiguous()
    assert not context.value_write.is_contiguous()
    assert context.key_write.data_ptr() == key.data_ptr()
    assert context.value_write.data_ptr() == value.data_ptr()


def test_paged_adapter_packs_noncontiguous_kv_head_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=5, seq_len=5)]
    )
    query = torch.randn(1, 5, 2, 4)
    key = torch.randn(1, 5, 4, 2).transpose(-1, -2)
    value = torch.randn(1, 5, 4, 2).transpose(-1, -2)
    assert key.shape == (1, 5, 2, 4)
    assert key.stride(-1) != 1

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, key, value)

    assert context.key_write.is_contiguous()
    assert context.value_write.is_contiguous()


def test_omni_paged_backend_defers_backend_owned_cache_update(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    layer.attn_backend.forward_includes_kv_cache_update = True
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    qkv = torch.randn(1, 3, 2, 4)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, qkv)
    assert layer.native_events == ["forward"]
    assert layer.updates == []


def test_omni_piecewise_backend_defers_backend_owned_cache_update(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, events = _make_adapter(monkeypatch)
    layer.attn_backend.forward_includes_kv_cache_update = True
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(full_attn_spans=[[(1, 3)]], extra={})

    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, qkv)
    assert layer.native_events == ["forward", "forward", "forward"]
    assert layer.updates == []
    assert [call[0].shape[0] for call in layer.calls] == [1, 2, 1]
    assert [call[1].shape[0] for call in layer.calls] == [1, 2, 1]
    assert [call[2].shape[0] for call in layer.calls] == [1, 2, 1]
    assert all(call[3] is event[2] for call, event in zip(layer.calls, events[1:], strict=True))
    segment_metadata = [call[3] for call in layer.calls]
    assert [metadata.causal for metadata in segment_metadata] == [True, False, True]
    assert [metadata.seq_lens.tolist() for metadata in segment_metadata] == [[1], [3], [4]]
    assert [metadata.query_start_loc_cpu.tolist() for metadata in segment_metadata] == [[0, 1], [0, 2], [0, 1]]
    assert [metadata.positions.tolist() for metadata in segment_metadata] == [[0], [1, 2], [3]]
    assert [metadata.slot_mappings.tolist() for metadata in segment_metadata] == [[[0]], [[1, 2]], [[3]]]
    for event in events[1:]:
        build_kwargs = event[1]
        assert build_kwargs["positions"].untyped_storage().data_ptr() == batch.positions.untyped_storage().data_ptr()
        assert (
            build_kwargs["slot_mappings"].untyped_storage().data_ptr()
            == batch.slot_mappings.untyped_storage().data_ptr()
        )


def test_platform_prewrite_updates_once_then_piecewise_reads_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    layer.attn_backend.forward_includes_kv_cache_update = True
    monkeypatch.setattr(
        adapter_module.current_omni_platform,
        "requires_diffusion_paged_kv_prewrite",
        lambda: True,
    )
    adapter.resolve_row = lambda *_args: DiffusionPagedAttentionRowBinding(
        row_index=2,
        max_seq_len=16,
        block_ids=((3, 7, 9, 11),),
    )
    row = DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)
    batch = adapter.prepare_batch([row])
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(full_attn_spans=[[(1, 3)]], extra={})

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", qkv, qkv, qkv, omni_attn_metadata=metadata)
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, qkv)
    assert layer.native_events == ["update", "forward", "forward", "forward"]
    assert len(layer.updates) == 1
    assert all(call[1] is None and call[2] is None for call in layer.calls)


def test_omni_paged_backend_runs_hunyuan_piecewise_segments(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, events = _make_adapter(monkeypatch)
    # Exercise the Ascend-only homogeneous batch path. CUDA keeps its native
    # output-buffer contract and is covered by the GPU adapter tests.
    monkeypatch.setattr(adapter_module.current_omni_platform, "is_npu", lambda: True)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=6,
                seq_len=9,
                kv_start_pos=3,
            ),
            DiffusionPagedAttentionRow(
                request_id="req-1",
                sequence_id=1,
                query_len=6,
                seq_len=9,
                kv_start_pos=3,
            ),
        ]
    )
    query = torch.arange(2 * 6 * 2 * 4, dtype=torch.float32).reshape(2, 6, 2, 4)
    key = torch.arange(2 * 6 * 2 * 4, dtype=torch.float32).reshape(2, 6, 2, 4) + 100
    value = key + 100
    attn_mask = torch.ones(9, 9, dtype=torch.bool).tril().repeat(2, 1, 1)
    attn_mask[:, 5:8, 5:8] = True
    metadata = SimpleNamespace(
        attn_mask=attn_mask.unsqueeze(1),
        full_attn_spans=[[(5, 8)], [(5, 8)]],
        query_ranges=None,
        extra={},
    )

    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            "layer-0",
            query,
            key,
            value,
            omni_attn_metadata=metadata,
        )
        # Every model layer sees the same spans, so the native segment metadata
        # is built once per active batch and then reused.
        adapter.prepare_layer_context(
            "layer-0",
            query,
            key,
            value,
            omni_attn_metadata=metadata,
        )
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, query)
    assert layer.native_events == ["update", "forward", "forward", "forward"]
    assert len(layer.updates) == 1
    assert layer.updates[0][0].shape[0] == 12
    assert torch.equal(layer.updates[0][0], key.reshape(12, 2, 4))
    assert torch.equal(layer.updates[0][1], value.reshape(12, 2, 4))
    assert [call[0].shape[0] for call in layer.calls] == [4, 6, 2]
    assert [call[1].shape[0] for call in layer.calls] == [4, 6, 2]
    assert [call[2].shape[0] for call in layer.calls] == [4, 6, 2]
    assert all(call[3] is event[2] for call, event in zip(layer.calls, events[1:], strict=True))
    assert [metadata.build_id for metadata in context.piecewise_native_metadata] == [1, 2, 3]

    # The first metadata build is the normal whole-query path. Piecewise calls
    # then use causal [3, 5), full [5, 8), and causal [8, 9) segments.
    assert len(events) == 4
    segment_builds = [event[1] for event in events[1:]]
    assert [build["causal"] for build in segment_builds] == [True, False, True]
    assert [build["seq_lens"].tolist() for build in segment_builds] == [
        [5, 5],
        [8, 8],
        [9, 9],
    ]
    assert [build["query_start_loc_cpu"].tolist() for build in segment_builds] == [
        [0, 2, 4],
        [0, 3, 6],
        [0, 1, 2],
    ]
    segment_metadata = [call[3] for call in layer.calls]
    assert [metadata.causal for metadata in segment_metadata] == [True, False, True]
    assert [metadata.seq_lens.tolist() for metadata in segment_metadata] == [[5, 5], [8, 8], [9, 9]]
    assert [metadata.query_start_loc_cpu.tolist() for metadata in segment_metadata] == [
        [0, 2, 4],
        [0, 3, 6],
        [0, 1, 2],
    ]
    assert [metadata.positions.tolist() for metadata in segment_metadata] == [
        [3, 4, 3, 4],
        [5, 6, 7, 5, 6, 7],
        [8, 8],
    ]
    assert [metadata.slot_mappings.tolist() for metadata in segment_metadata] == [
        [[3, 4, 3, 4]],
        [[5, 6, 7, 5, 6, 7]],
        [[8, 8]],
    ]


def test_piecewise_metadata_snapshots_reused_native_scheduler_buffer(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    shared_scheduler_metadata = torch.zeros(1, dtype=torch.int32)
    build_id = 0

    def build_metadata(**kwargs):
        nonlocal build_id
        shared_scheduler_metadata.fill_(build_id)
        metadata = SimpleNamespace(
            build_id=build_id,
            causal=kwargs["causal"],
            seq_lens=kwargs["seq_lens"].clone(),
            query_start_loc_cpu=kwargs["query_start_loc_cpu"].clone(),
            positions=kwargs["positions"].clone(),
            slot_mappings=kwargs["slot_mappings"].clone(),
            scheduler_metadata=shared_scheduler_metadata,
            prefix_scheduler_metadata=None,
        )
        build_id += 1
        return {"layer-0": metadata}

    monkeypatch.setattr(adapter_module.current_omni_platform, "build_diffusion_kv_attn_metadata", build_metadata)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(full_attn_spans=[[(1, 3)]], extra={})

    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    segment_schedules = [item.scheduler_metadata for item in context.piecewise_native_metadata]
    assert [item.item() for item in segment_schedules] == [1, 2, 3]
    assert len({item.data_ptr() for item in segment_schedules}) == 3


def test_omni_paged_backend_treats_empty_full_spans_as_causal(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, events = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(
        attn_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool).tril(),
        full_attn_spans=[[]],
        query_ranges=None,
        extra={},
    )

    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, qkv)
    assert layer.native_events == ["update", "forward"]
    assert len(events) == 2
    assert events[1][1]["causal"] is True
    assert events[1][1]["seq_lens"].tolist() == [4]


def test_omni_paged_backend_runs_unequal_hunyuan_prefixes_without_padding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, layer, events = _make_adapter(monkeypatch, capacity=64)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=17,
                seq_len=50,
                kv_start_pos=33,
            ),
            DiffusionPagedAttentionRow(
                request_id="req-1",
                sequence_id=1,
                query_len=17,
                seq_len=54,
                kv_start_pos=37,
            ),
        ]
    )
    query = torch.arange(2 * 17 * 2 * 4, dtype=torch.float32).reshape(2, 17, 2, 4)
    key = query + 100
    value = query + 200
    metadata = SimpleNamespace(
        attn_mask=torch.ones(2, 1, 17, 54, dtype=torch.bool),
        full_attn_spans=[[(33, 50)], [(37, 54)]],
        query_ranges=None,
        extra={},
    )

    with adapter.activate(batch):
        context = adapter.prepare_layer_context(
            "layer-0",
            query,
            key,
            value,
            omni_attn_metadata=metadata,
        )
        output = _run_omni_paged_backend(context)

    assert torch.equal(output, query)
    assert batch.query_start_loc.tolist() == [0, 17, 34]
    assert batch.seq_lens.tolist() == [50, 54]
    assert batch.positions.tolist() == [*range(33, 50), *range(37, 54)]
    assert torch.equal(context.key_write, key.reshape(34, 2, 4))
    assert torch.equal(context.value_write, value.reshape(34, 2, 4))
    assert layer.native_events == ["update", "forward"]
    assert layer.updates[0][2].tolist() == batch.positions.tolist()
    assert layer.calls[0][3].seq_lens.tolist() == [50, 54]
    assert layer.calls[0][3].query_start_loc_cpu.tolist() == [0, 17, 34]
    assert len(events) == 2


def test_forward_rejects_unaligned_heterogeneous_piecewise_spans_before_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4),
            DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=4, seq_len=4),
        ]
    )
    qkv = torch.randn(2, 4, 2, 4)
    metadata = SimpleNamespace(
        attn_mask=torch.ones(2, 1, 4, 4, dtype=torch.bool),
        full_attn_spans=[[(1, 3)], [(2, 4)]],
        query_ranges=None,
        extra={},
    )

    with adapter.activate(batch), pytest.raises(ValueError, match="produce aligned segments"):
        adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    assert layer.native_events == []


@pytest.mark.parametrize(
    "spans",
    [
        [(1, 3), (2, 4)],
        [(1, 5)],
    ],
)
def test_forward_rejects_invalid_piecewise_spans_before_cache_write(
    monkeypatch: pytest.MonkeyPatch,
    spans: list[tuple[int, int]],
) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(full_attn_spans=[spans], extra={})

    with adapter.activate(batch), pytest.raises(ValueError, match="sorted, non-overlapping, and within the sequence"):
        adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    assert layer.native_events == []


def test_forward_rejects_non_4d_mask_with_piecewise_spans_before_cache_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(
        attn_mask=torch.ones(1, 4, 4, dtype=torch.bool),
        full_attn_spans=[[(1, 3)]],
        extra={},
    )

    with adapter.activate(batch), pytest.raises(NotImplementedError, match="requires a 4D tensor"):
        adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    assert layer.native_events == []


def test_forward_rejects_piecewise_query_ranges_before_cache_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    qkv = torch.randn(1, 4, 2, 4)
    metadata = SimpleNamespace(
        attn_mask=torch.ones(1, 1, 4, 4, dtype=torch.bool),
        full_attn_spans=[[(1, 3)]],
        query_ranges=(object(),),
        extra={},
    )

    with adapter.activate(batch), pytest.raises(NotImplementedError, match="query_ranges"):
        adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    assert layer.native_events == []


def test_forward_rejects_piecewise_non_suffix_query_before_cache_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=6,
                kv_start_pos=2,
            )
        ]
    )
    qkv = torch.randn(1, 2, 2, 4)
    metadata = SimpleNamespace(full_attn_spans=[[(2, 4)]], extra={})

    with adapter.activate(batch), pytest.raises(ValueError, match="requires each query/write span to end"):
        adapter.prepare_layer_context(
            "layer-0",
            qkv,
            qkv,
            qkv,
            omni_attn_metadata=metadata,
        )

    assert layer.native_events == []


def test_forward_accepts_non_aligned_current_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=5,
                kv_start_pos=3,
            )
        ]
    )
    query = torch.randn(1, 2, 2, 4)
    key = torch.arange(1 * 2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4)
    value = key + 100

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, key, value)

    assert context.query.shape == (2, 2, 4)
    assert torch.equal(context.key_write, key.reshape(2, 2, 4))
    assert torch.equal(context.value_write, value.reshape(2, 2, 4))
    assert context.slot_mapping.tolist() == [3, 4]
    assert layer.native_events == []


def test_forward_rejects_untranslated_omni_metadata_before_cache_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, layer, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    qkv = torch.randn(1, 2, 4)
    metadata = SimpleNamespace(attn_mask=torch.ones(1, 1), extra={})

    with adapter.activate(batch), pytest.raises(NotImplementedError, match="cannot translate"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv, omni_attn_metadata=metadata)

    assert layer.updates == []


def test_forward_accepts_flattened_hidden_dimension(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    query = torch.randn(3, 8)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", query, query, query)

    assert context.query.shape == (3, 2, 4)
    assert context.restore_output(context.query).shape == query.shape


def test_forward_accepts_one_flattened_token_for_single_head(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch, num_heads=1, num_kv_heads=1)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    qkv = torch.randn(1, 4)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)
        output = _run_omni_paged_backend(context)

    assert context.query.shape == (1, 1, 4)
    assert context.query_token_shape == (1,)
    assert not context.query_has_head_dims
    assert output.shape == qkv.shape


def test_forward_requires_active_prepared_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    qkv = torch.randn(1, 2, 4)

    with pytest.raises(RuntimeError, match="adapter.activate"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)


def test_new_preparation_invalidates_native_buffer_views_in_older_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    old_batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    adapter.prepare_batch([DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1)])

    with pytest.raises(ValueError, match="stale"):
        with adapter.activate(old_batch):
            pass


def test_block_table_change_invalidates_prepared_native_buffer_views(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    adapter.invalidate_prepared_batches()

    with pytest.raises(ValueError, match="stale"):
        with adapter.activate(batch):
            pass


def test_block_table_change_is_rejected_during_active_forward(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    with adapter.activate(batch), pytest.raises(RuntimeError, match="during an active forward"):
        adapter.invalidate_prepared_batches()


def test_prepare_batch_rejects_active_native_buffers(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )

    with adapter.activate(batch), pytest.raises(RuntimeError, match="during an active forward"):
        adapter.prepare_batch([DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1)])


def test_forward_rejects_tensor_count_different_from_prepared_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=3, seq_len=3)]
    )
    qkv = torch.randn(2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="token count"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)


def test_forward_rejects_dtype_different_from_model_activation_dtype(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1)]
    )
    qkv = torch.randn(1, 2, 4, dtype=torch.float64)

    with adapter.activate(batch), pytest.raises(ValueError, match="model activation dtype"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)


def test_forward_rejects_invalid_key_token_count(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=4, seq_len=4)]
    )
    query = torch.randn(4, 2, 4)
    key = torch.randn(3, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="key token count"):
        adapter.prepare_layer_context("layer-0", query, key, query)


def test_forward_rejects_batched_layout_that_does_not_match_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1),
            DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=3, seq_len=3),
        ]
    )
    qkv = torch.randn(2, 2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="must match prepared rows"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)


def test_forward_packs_and_restores_right_padded_heterogeneous_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=1, seq_len=1),
            DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=3, seq_len=3),
        ]
    )
    qkv = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)

    with adapter.activate(batch):
        context = adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)

    assert context.output_scatter_indices.tolist() == [0, 3, 4, 5]
    torch.testing.assert_close(context.query, qkv.reshape(-1, 2, 4)[[0, 3, 4, 5]])
    restored = context.restore_output(context.query)
    assert restored.shape == qkv.shape
    torch.testing.assert_close(restored[0, 0], qkv[0, 0])
    assert torch.count_nonzero(restored[0, 1:]) == 0
    torch.testing.assert_close(restored[1], qkv[1])


def test_forward_rejects_padded_full_kv(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    batch = adapter.prepare_batch(
        [
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=1,
                seq_len=3,
                kv_start_pos=2,
            ),
            DiffusionPagedAttentionRow(
                request_id="req-1",
                sequence_id=1,
                query_len=1,
                seq_len=5,
                kv_start_pos=4,
            ),
        ]
    )
    query = torch.randn(2, 1, 2, 4)
    full_kv = torch.randn(2, 4, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="batched key layout must match prepared rows"):
        adapter.prepare_layer_context("layer-0", query, full_kv, full_kv)


def test_runner_runtime_switches_from_prefill_to_reusable_denoise_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch)
    metadata = DiffusionPagedAttentionMetadata(
        prefill_rows=(DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=6, seq_len=6),),
        denoise_rows=(
            DiffusionPagedAttentionRow(
                request_id="req-0",
                sequence_id=0,
                query_len=2,
                seq_len=5,
                kv_start_pos=3,
            ),
        ),
    )
    runtime = DiffusionPagedAttentionRuntime(adapter, metadata)

    with set_forward_context(paged_kv_runtime=runtime), runtime.activate():
        set_forward_context_denoise_step_idx(0)
        assert Attention._active_paged_kv_adapter() is adapter
        assert adapter._active_batch.rows == metadata.prefill_rows

        set_forward_context_denoise_step_idx(1)
        assert Attention._active_paged_kv_adapter() is adapter
        denoise_batch = adapter._active_batch
        assert denoise_batch.rows == metadata.denoise_rows

        set_forward_context_denoise_step_idx(2)
        assert Attention._active_paged_kv_adapter() is adapter
        assert adapter._active_batch is denoise_batch

    assert adapter._active_batch is None


def test_causal_batch_rejects_non_suffix_write(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, _, _, _ = _make_adapter(monkeypatch, non_causal=False)
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=2,
        seq_len=8,
        kv_start_pos=3,
    )

    batch = adapter.prepare_batch([row])
    qkv = torch.randn(2, 2, 4)

    with adapter.activate(batch), pytest.raises(ValueError, match="requires each query/write span to end"):
        adapter.prepare_layer_context("layer-0", qkv, qkv, qkv)


def test_prepare_batch_rejects_native_decode_after_prefill_order(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter, block_tables, _, _ = _make_adapter(
        monkeypatch,
        non_causal=False,
        reorder_batch_threshold=1,
    )
    rows = [
        DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=2, seq_len=2),
        DiffusionPagedAttentionRow(request_id="req-1", sequence_id=1, query_len=1, seq_len=1),
    ]

    with pytest.raises(ValueError, match="decode/short-query rows before"):
        adapter.prepare_batch(rows)

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_rejects_seq_len_beyond_installed_row_capacity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, _ = _make_adapter(monkeypatch)
    block_tables.num_blocks.np[0, 2] = 1
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=5,
        seq_len=5,
    )

    with pytest.raises(ValueError, match="installed Worker row contains only 1 blocks"):
        adapter.prepare_batch([row])

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_rejects_seq_len_beyond_installed_logical_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, _ = _make_adapter(monkeypatch)
    adapter.resolve_row = lambda *_args: DiffusionPagedAttentionRowBinding(2, 4)
    row = DiffusionPagedAttentionRow(
        request_id="req-0",
        sequence_id=0,
        query_len=5,
        seq_len=5,
    )

    with pytest.raises(ValueError, match="installed allocation has logical length 4"):
        adapter.prepare_batch([row])

    assert block_tables.gather_calls == []
    assert block_tables.slot_calls == []


def test_prepare_batch_passes_dcp_local_seq_lens_to_native_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, block_tables, _, events = _make_adapter(monkeypatch)
    block_tables.cp_size = 2
    block_tables.cp_rank = 1
    block_tables.cp_interleave = 2
    block_tables.num_blocks.np[0, 2] = 1

    batch = adapter.prepare_batch(
        [DiffusionPagedAttentionRow(request_id="req-0", sequence_id=0, query_len=5, seq_len=5)]
    )

    assert batch.row_indices.tolist() == [2]
    assert events[0][1]["dcp_local_seq_lens"].tolist() == [2]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"sequence_id": 0, "context_id": "text"}, "exactly one"),
        ({}, "exactly one"),
        ({"sequence_id": 0, "query_len": 0}, "query_len"),
        ({"sequence_id": 0, "kv_start_pos": 4, "query_len": 2, "seq_len": 5}, "exceeds seq_len"),
    ],
)
def test_row_contract_rejects_invalid_identity_or_span(kwargs: dict, message: str) -> None:
    values = {
        "request_id": "req-0",
        "query_len": 1,
        "seq_len": 1,
        **kwargs,
    }
    with pytest.raises(ValueError, match=message):
        DiffusionPagedAttentionRow(**values)


def test_layer_adapter_accepts_platform_native_backend_and_uses_rank_local_heads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    selected_backends = []
    specialized_backends = []
    impl_cls = Mock(return_value=SimpleNamespace(forward=Mock(), do_kv_cache_update=Mock()))
    native_backend = SimpleNamespace(
        get_name=lambda: "ASCEND",
        indexes_kv_by_block_stride=lambda: True,
        get_impl_cls=lambda: impl_cls,
        forward_includes_kv_cache_update=True,
    )

    original_backend_per_kind = {"full": object()}
    config = SimpleNamespace(
        attention_config=SimpleNamespace(backend=None, backend_per_kind=original_backend_per_kind),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=2),
        model_config=SimpleNamespace(dtype=torch.float16),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )

    def select_backend(**_kwargs):
        selected_backends.append(
            (
                config.attention_config.backend,
                config.attention_config.backend_per_kind,
                config.parallel_config.prefill_context_parallel_size,
            )
        )
        return native_backend

    monkeypatch.setattr(adapter_module, "get_attn_backend", select_backend)
    monkeypatch.setattr(adapter_module, "set_current_vllm_config", lambda _config: nullcontext())
    monkeypatch.setattr(
        adapter_module.current_omni_platform,
        "get_diffusion_paged_kv_attn_backend",
        lambda backend, *, ulysses_degree: specialized_backends.append((backend, ulysses_degree)) or backend,
    )
    layer = SimpleNamespace(num_heads=8, softmax_scale=0.125)
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float16,
        non_causal=True,
    )

    native_layer = adapter_module.DiffusionPagedAttentionLayerAdapter(
        layer_name="layer-0",
        layer=layer,
        spec=spec,
        vllm_config=config,
        device=torch.device("cpu"),
        ulysses_degree=2,
    )

    assert selected_backends == [(adapter_module.AttentionBackendEnum.FLASH_ATTN, {}, 2)]
    assert specialized_backends == [(native_backend, 2)]
    assert config.attention_config.backend is None
    assert config.attention_config.backend_per_kind is original_backend_per_kind
    assert config.parallel_config.prefill_context_parallel_size == 2
    assert native_layer.num_heads == 4
    assert native_layer.num_kv_heads == 2
    assert native_layer.spec.num_kv_heads == 2
    assert native_layer.spec.indexes_kv_by_block_stride is True
    assert native_layer._q_scale_float == 1.0
    assert native_layer._k_scale_float == 1.0
    assert native_layer._v_scale_float == 1.0
    impl_kwargs = impl_cls.call_args.kwargs
    assert impl_kwargs["num_heads"] == 4
    assert impl_kwargs["num_kv_heads"] == 2


def test_omni_attention_wraps_paged_kernel_with_sp_hooks() -> None:
    events: list[str] = []

    class Strategy:
        name = "ulysses"

        def pre_attention(self, query, key, value, metadata):
            events.append("pre")
            return query + 1, key + 2, value + 3, metadata, object()

        def post_attention(self, output, _ctx):
            events.append("post")
            return output + 4

    class Adapter:
        def prepare_layer_context(self, layer_name, query, key, value, *, omni_attn_metadata):
            events.append("prepare")
            assert layer_name == "layer-0"
            assert omni_attn_metadata is None
            assert torch.equal(key, original + 2)
            assert torch.equal(value, original + 3)
            return SimpleNamespace(query=query)

    class Backend:
        def forward_paged(self, context):
            events.append("kernel")
            return context.query * 2

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.prefix = "layer-0"
    layer.paged_kv_cache_role = "primary"
    layer.attn_backend = SimpleNamespace(supports_paged_kv=True, get_name=lambda: "FLASH_ATTN")
    layer.attention = Backend()
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: Strategy()
    layer._with_kv_cache_dtype = lambda metadata: metadata
    original = torch.zeros(1, 2, 2, 4)

    with set_forward_context(), override_paged_kv_adapter(Adapter()):
        assert layer.is_paged_kv_active()
        output = layer._forward_impl(original, original, original)

    assert events == ["pre", "prepare", "kernel", "post"]
    assert torch.equal(output, torch.full_like(original, 6))


def test_omni_attention_strips_paged_ulysses_padding_around_kernel() -> None:
    events: list[tuple[str, tuple[int, ...]]] = []

    class Strategy:
        name = "ulysses"

        def pre_attention(self, query, key, value, metadata):
            return query, key, value, metadata, SimpleNamespace(joint_len=0, joint_strategy="front")

        def post_attention(self, output, _ctx):
            events.append(("post", tuple(output.shape)))
            return output

    class Adapter:
        def prepare_layer_context(self, _layer_name, query, key, value, *, omni_attn_metadata):
            assert omni_attn_metadata is None
            events.append(("kernel_input", tuple(query.shape)))
            assert query[:, -1].tolist() == [[[3.0]]]
            assert torch.equal(query, key)
            assert torch.equal(query, value)
            return SimpleNamespace(query=query)

    class Backend:
        def forward_paged(self, context):
            return context.query

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.prefix = "layer-0"
    layer.paged_kv_cache_role = "primary"
    layer.attn_backend = SimpleNamespace(supports_paged_kv=True, get_name=lambda: "FLASH_ATTN")
    layer.attention = Backend()
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: Strategy()
    tensor = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1, 1)

    with set_forward_context(), override_paged_kv_adapter(Adapter()):
        get_forward_context().sp_padding_size = 1
        output = layer._forward_impl(tensor, tensor, tensor)

    assert events == [("kernel_input", (1, 4, 1, 1)), ("post", (1, 5, 1, 1))]
    assert output.flatten().tolist() == [0.0, 1.0, 2.0, 3.0, 0.0]


def test_omni_attention_rejects_paged_request_for_backend_without_paged_support() -> None:
    class Adapter:
        def prepare_layer_context(self, *args, **kwargs):
            raise AssertionError("an unsupported backend must fail before adapter preparation")

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.prefix = "layer-0"
    layer.paged_kv_cache_role = "primary"
    layer.attn_backend = SimpleNamespace(supports_paged_kv=False, get_name=lambda: "SDPA")
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: layer._no_parallel_strategy

    with (
        set_forward_context(),
        override_paged_kv_adapter(Adapter()),
        pytest.raises(NotImplementedError, match="backend with paged support"),
    ):
        layer._forward_impl(torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4), torch.zeros(1, 2, 2, 4))


def test_omni_attention_keeps_dense_kernel_without_active_adapter() -> None:
    events: list[str] = []

    class Strategy:
        name = "ulysses"

        def pre_attention(self, query, key, value, metadata):
            events.append("pre")
            return query, key, value, metadata, object()

        def post_attention(self, output, _ctx):
            events.append("post")
            return output

    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.paged_kv_cache_role = "primary"
    layer.use_ring = False
    layer._no_parallel_strategy = object()
    layer._get_active_parallel_strategy = lambda: Strategy()
    layer._with_kv_cache_dtype = lambda metadata: metadata
    layer._run_local_attention = lambda query, _key, _value, _metadata: events.append("dense") or query
    qkv = torch.zeros(1, 2, 2, 4)

    assert not layer.is_paged_kv_active()
    output = layer._forward_impl(qkv, qkv, qkv)

    assert events == ["pre", "dense", "post"]
    assert output is qkv


def test_omni_attention_rejects_scheduler_paged_forward_without_active_adapter() -> None:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._scheduler_paged_kv = True
    layer.paged_kv_cache_role = "primary"
    layer._get_active_parallel_strategy = lambda: object()
    qkv = torch.zeros(1, 2, 2, 4)

    with set_forward_context(), pytest.raises(RuntimeError, match="without an active Worker adapter"):
        layer._forward_impl(qkv, qkv, qkv)
