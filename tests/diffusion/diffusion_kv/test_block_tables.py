# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from vllm.v1.kv_cache_interface import FullAttentionSpec
from vllm.v1.worker import block_table as native_block_table

from vllm_omni.diffusion.diffusion_kv import model_runner_backend as model_runner_backend_module
from vllm_omni.diffusion.diffusion_kv.config import DiffusionKVCacheMode
from vllm_omni.diffusion.diffusion_kv.metadata import (
    DiffusionKVContextMetadata,
    DiffusionKVMetadata,
    DiffusionKVSequenceMetadata,
)
from vllm_omni.diffusion.diffusion_kv.model_runner_backend import DiffusionKVModelRunnerBackend
from vllm_omni.diffusion.sched.interface import CachedRequestData, DiffusionSchedulerOutput
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _registration_backend(parallel_config: object):
    model_config = SimpleNamespace(set_attention_geometry=Mock())
    vllm_config = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        compilation_config=SimpleNamespace(static_forward_context={}),
        model_config=model_config,
    )
    backend = DiffusionKVModelRunnerBackend(
        vllm_config=vllm_config,
        od_config=SimpleNamespace(parallel_config=parallel_config),
        device=torch.device("cpu"),
    )
    return backend, vllm_config, model_config


def _parallel_config(**overrides):
    values = dict(ulysses_degree=1, ring_degree=1, allgather_degree=1, ulysses_mode="strict")
    values.update(overrides)
    return SimpleNamespace(**values)


def test_cache_layer_registration_installs_adapters_with_ulysses_local_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend, vllm_config, model_config = _registration_backend(_parallel_config(ulysses_degree=2))
    layer = SimpleNamespace(
        num_heads=8,
        softmax_scale=0.125,
        skip_sequence_parallel=False,
        attn_backend=SimpleNamespace(supports_paged_kv=True, get_name=lambda: "FLASH_ATTN"),
    )
    unrelated = object()
    vllm_config.compilation_config.static_forward_context.update({"layer-0": layer, "unrelated": unrelated})
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float16,
        non_causal=True,
    )
    constructor_calls = []

    class FakeLayerAdapter:
        def __init__(self, *, layer_name, layer, spec, ulysses_degree, **_kwargs) -> None:
            constructor_calls.append((layer_name, layer, spec, ulysses_degree))
            self.num_heads = layer.num_heads // ulysses_degree
            self.num_kv_heads = spec.num_kv_heads // ulysses_degree
            self.head_size = spec.head_size
            self.spec = replace(spec, num_kv_heads=self.num_kv_heads)

    monkeypatch.setattr(model_runner_backend_module, "DiffusionPagedAttentionLayerAdapter", FakeLayerAdapter)
    monkeypatch.setattr(model_runner_backend_module, "set_current_vllm_config", lambda _config: nullcontext())

    specs = backend.register_kv_cache_layers({"layer-0": (layer, spec)})

    installed = vllm_config.compilation_config.static_forward_context["layer-0"]
    assert isinstance(installed, FakeLayerAdapter)
    assert vllm_config.compilation_config.static_forward_context["unrelated"] is unrelated
    assert specs["layer-0"].num_kv_heads == 2
    assert constructor_calls == [("layer-0", layer, spec, 2)]
    model_config.set_attention_geometry.assert_called_once_with(
        num_heads=4,
        num_kv_heads=2,
        head_size=8,
    )
    assert vllm_config.attention_config.use_non_causal is True


def test_cache_layer_registration_rejects_backend_without_paged_support() -> None:
    backend, _, _ = _registration_backend(_parallel_config())
    layer = SimpleNamespace(
        attn_backend=SimpleNamespace(supports_paged_kv=False, get_name=lambda: "SDPA"),
    )
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=4,
        head_size=8,
        dtype=torch.float16,
    )

    with pytest.raises(NotImplementedError, match="layer-0.*paged support.*SDPA"):
        backend.register_kv_cache_layers({"layer-0": (layer, spec)})


def test_cache_layer_registration_resolves_platform_paged_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    backend, _, _ = _registration_backend(_parallel_config())

    def unsupported_block_tables():
        raise NotImplementedError("platform has no paged BlockTables")

    monkeypatch.setattr(
        model_runner_backend_module.current_omni_platform,
        "get_diffusion_kv_block_tables_cls",
        unsupported_block_tables,
    )

    with pytest.raises(NotImplementedError, match="no paged BlockTables"):
        backend.register_kv_cache_layers({"layer-0": (object(), Mock(spec=FullAttentionSpec))})


@pytest.mark.parametrize(
    ("parallel_config", "message"),
    [
        (_parallel_config(ring_degree=2), "Ring"),
        (_parallel_config(allgather_degree=2), "AllGather-KV"),
        (_parallel_config(ulysses_degree=2, ulysses_mode="advanced_uaa"), "strict Ulysses"),
    ],
)
def test_cache_layer_registration_rejects_unsupported_sp_modes(parallel_config, message: str) -> None:
    backend, _, _ = _registration_backend(parallel_config)

    with pytest.raises(NotImplementedError, match=message):
        backend._get_paged_attention_ulysses_degree(SimpleNamespace(skip_sequence_parallel=False))


def test_cache_layer_registration_ignores_sp_for_opted_out_layer() -> None:
    backend, _, _ = _registration_backend(
        _parallel_config(ulysses_degree=2, ring_degree=2, ulysses_mode="advanced_uaa")
    )

    assert backend._get_paged_attention_ulysses_degree(SimpleNamespace(skip_sequence_parallel=True)) == 1


class _FakeSpec:
    def __init__(self, block_size: int) -> None:
        self.block_size = block_size
        self.capacity_calls: list[tuple[object, int]] = []

    def max_num_blocks_per_req(self, vllm_config: object, max_len: int) -> int:
        self.capacity_calls.append((vllm_config, max_len))
        return math.ceil(max_len / self.block_size)


class _FakeStagedTable:
    def __init__(self) -> None:
        self.staged: list[tuple[int, tuple[int, ...]]] = []
        self.clear_calls = 0

    def clear_staged_writes(self) -> None:
        self.staged.clear()
        self.clear_calls += 1


class _FakeNumBlocks:
    def __init__(self, num_groups: int, max_num_reqs: int) -> None:
        self.np = np.zeros((num_groups, max_num_reqs), dtype=np.int32)
        self.copy_calls = 0

    def copy_to_uva(self) -> None:
        self.copy_calls += 1


class _FakeBlockTables:
    instances: list[_FakeBlockTables] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.block_sizes = list(kwargs["block_sizes"])
        self.max_num_reqs = int(kwargs["max_num_reqs"])
        self.max_num_batched_tokens = int(kwargs["max_num_batched_tokens"])
        self.cp_size = int(kwargs["cp_size"])
        self.cp_rank = int(kwargs["cp_rank"])
        self.cp_interleave = int(kwargs["cp_interleave"])
        self.blocks_per_kv_block = [1] * len(self.block_sizes)
        self.block_tables = [_FakeStagedTable() for _ in self.block_sizes]
        self.num_blocks = _FakeNumBlocks(len(self.block_sizes), self.max_num_reqs)
        self.append_calls: list[tuple[int, tuple[list[int], ...], bool]] = []
        self.apply_calls = 0
        self.layout_refreshes = 0
        self.fail_on_append_call: int | None = None
        self.fail_apply = False
        self.committed: list[dict[int, tuple[int, ...]]] = [dict() for _ in self.block_sizes]
        self.__class__.instances.append(self)

    def append_block_ids(
        self,
        req_index: int,
        new_block_ids: tuple[list[int], ...],
        overwrite: bool,
    ) -> None:
        call_number = len(self.append_calls) + 1
        if self.fail_on_append_call == call_number:
            raise RuntimeError("fake append failure")
        copied_ids = tuple(list(group_ids) for group_ids in new_block_ids)
        self.append_calls.append((req_index, copied_ids, overwrite))
        for group_index, group_ids in enumerate(copied_ids):
            self.num_blocks.np[group_index, req_index] = len(group_ids)
            if group_ids:
                self.block_tables[group_index].staged.append((req_index, tuple(group_ids)))

    def apply_staged_writes(self) -> None:
        self.apply_calls += 1
        if self.fail_apply:
            raise RuntimeError("fake apply failure")
        for group_index, group_table in enumerate(self.block_tables):
            for row, group_ids in group_table.staged:
                self.committed[group_index][row] = group_ids
            group_table.staged.clear()
        self.num_blocks.copy_to_uva()

    def init_block_table_layout_tensors(self) -> None:
        self.layout_refreshes += 1


def _metadata(
    request_id: str = "req-0",
    generation: int = 1,
    *,
    seq_len: int = 9,
    sequence_block_ids: tuple[list[int], ...] = ([1, 2, 3], [4, 5]),
    contexts: tuple[DiffusionKVContextMetadata, ...] | None = None,
) -> DiffusionKVMetadata:
    if contexts is None:
        contexts = (
            DiffusionKVContextMetadata(
                context_id="text",
                cache_role="cross_attention",
                num_tokens=5,
                block_ids=([6, 7], [8]),
            ),
        )
    return DiffusionKVMetadata(
        request_id=request_id,
        allocation_generation=generation,
        sequences=(
            DiffusionKVSequenceMetadata(
                sequence_id=0,
                prefix_len=4,
                target_len=5,
                seq_len=seq_len,
                block_ids=sequence_block_ids,
                context_ids=tuple(context.context_id for context in contexts),
            ),
        ),
        contexts=contexts,
    )


def _install_native_modules(monkeypatch: pytest.MonkeyPatch, events: list[tuple]) -> None:
    _FakeBlockTables.instances.clear()

    def get_block_table_width(max_num_blocks, block_size):
        events.append(("width", max_num_blocks, block_size))
        return max_num_blocks + 1

    def init_backend(kv_cache_config, vllm_config, device):
        events.append(("backend", kv_cache_config, vllm_config, device))
        group = SimpleNamespace(get_metadata_builder=lambda _index: SimpleNamespace(reorder_batch_threshold=None))
        return [[group], [group]], "cg-support", [4, 8]

    def init_cache(
        runner_kv_caches,
        forward_context,
        kv_cache_config,
        attn_groups,
        device,
        cache_dtype,
        kernel_block_sizes,
        vllm_config,
    ):
        events.append(("cache", kv_cache_config, attn_groups, kernel_block_sizes))
        runner_kv_caches.extend(["cache-0", "cache-1"])
        return {"layer-0": "tensor-0", "layer-1": "tensor-1"}

    monkeypatch.setattr(model_runner_backend_module, "init_attn_backend", init_backend)
    monkeypatch.setattr(model_runner_backend_module, "init_kv_cache", init_cache)
    monkeypatch.setattr(
        model_runner_backend_module.current_omni_platform,
        "get_diffusion_kv_block_tables_cls",
        lambda: _FakeBlockTables,
    )
    monkeypatch.setattr(native_block_table, "get_block_table_width", get_block_table_width, raising=False)


def _runner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_rows_per_request: int = 3,
    max_num_seqs: int = 2,
    max_len: int = 16,
    max_num_batched_tokens: int = 32,
    cp_size: int = 1,
    cp_rank: int = 0,
    cp_interleave: int = 1,
) -> tuple[DiffusionKVModelRunnerBackend, _FakeBlockTables, list[tuple]]:
    events: list[tuple] = []
    _install_native_modules(monkeypatch, events)
    od_config = SimpleNamespace(
        diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
        diffusion_kv_max_rows_per_request=max_rows_per_request,
        max_num_seqs=max_num_seqs,
    )
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=max_len),
        scheduler_config=SimpleNamespace(
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=cp_size,
            cp_kv_cache_interleave_size=cp_interleave,
        ),
        compilation_config=SimpleNamespace(static_forward_context={"layer-0": object(), "layer-1": object()}),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    monkeypatch.setattr(
        model_runner_backend_module,
        "get_dcp_group",
        lambda: SimpleNamespace(rank_in_group=cp_rank),
    )
    backend = DiffusionKVModelRunnerBackend(
        vllm_config=vllm_config,
        od_config=od_config,
        device="cuda:1",
    )
    backend._kv_cache_layer_adapters = {
        "layer-0": SimpleNamespace(spec=SimpleNamespace(non_causal=False), kv_cache=None),
        "layer-1": SimpleNamespace(spec=SimpleNamespace(non_causal=False), kv_cache=None),
    }

    config = SimpleNamespace(
        num_blocks=32,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer-0"], kv_cache_spec=_FakeSpec(4)),
            SimpleNamespace(layer_names=["layer-1"], kv_cache_spec=_FakeSpec(8)),
        ],
    )
    backend.initialize_kv_cache(config)
    return backend, _FakeBlockTables.instances[-1], events


def test_initialize_builds_native_block_tables_from_rank_local_config(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, events = _runner(monkeypatch)

    assert events[:2] == [("width", 4, 4), ("width", 2, 8)]
    assert [event[0] for event in events[2:]] == ["backend", "cache"]
    assert block_tables.kwargs == {
        "block_sizes": [4, 8],
        "max_num_reqs": 6,
        "max_num_batched_tokens": 32,
        "max_num_blocks_per_group": [5, 3],
        "device": "cuda:1",
        "kernel_block_sizes": [4, 8],
        "cp_size": 1,
        "cp_rank": 0,
        "cp_interleave": 1,
    }
    assert runner.block_tables is block_tables
    assert runner._diffusion_kv_free_rows == [5, 4, 3, 2, 1, 0]
    assert runner.kv_caches == ["cache-0", "cache-1"]
    assert runner.vllm_config.scheduler_config.max_num_seqs == 6


def test_initialize_propagates_native_context_parallel_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    _, block_tables, _ = _runner(
        monkeypatch,
        cp_size=2,
        cp_rank=1,
        cp_interleave=4,
    )

    assert block_tables.kwargs["cp_size"] == 2
    assert block_tables.kwargs["cp_rank"] == 1
    assert block_tables.kwargs["cp_interleave"] == 4


def test_initialize_failure_does_not_publish_partial_cache_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple] = []
    _install_native_modules(monkeypatch, events)
    successful_init_cache = model_runner_backend_module.init_kv_cache
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(max_model_len=16),
        scheduler_config=SimpleNamespace(max_num_batched_tokens=32, max_num_seqs=1),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=1,
            cp_kv_cache_interleave_size=1,
        ),
        compilation_config=SimpleNamespace(static_forward_context={"layer-0": object(), "layer-1": object()}),
        cache_config=SimpleNamespace(cache_dtype="auto"),
    )
    backend = DiffusionKVModelRunnerBackend(
        vllm_config=vllm_config,
        od_config=SimpleNamespace(
            diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER,
            diffusion_kv_max_rows_per_request=2,
        ),
        device="cuda:1",
    )
    backend._kv_cache_layer_adapters = {
        "layer-0": SimpleNamespace(spec=SimpleNamespace(non_causal=False), kv_cache="placeholder-0"),
        "layer-1": SimpleNamespace(spec=SimpleNamespace(non_causal=False), kv_cache="placeholder-1"),
    }
    config = SimpleNamespace(
        num_blocks=32,
        kv_cache_groups=[
            SimpleNamespace(layer_names=["layer-0"], kv_cache_spec=_FakeSpec(4)),
            SimpleNamespace(layer_names=["layer-1"], kv_cache_spec=_FakeSpec(8)),
        ],
    )

    def fail_init_cache(runner_kv_caches, *_args, **_kwargs):
        runner_kv_caches.append("partial-cache")
        backend._kv_cache_layer_adapters["layer-0"].kv_cache = "partial-binding"
        raise RuntimeError("cache binding failed")

    monkeypatch.setattr(model_runner_backend_module, "init_kv_cache", fail_init_cache)
    with pytest.raises(RuntimeError, match="cache binding failed"):
        backend.initialize_kv_cache(config)

    assert backend.kv_cache_config is None
    assert backend.kv_caches == []
    assert backend.block_tables is None
    assert backend.paged_attention_adapter is None
    assert backend._kv_cache_layer_adapters["layer-0"].kv_cache == "placeholder-0"
    assert backend.vllm_config.scheduler_config.max_num_seqs == 1

    monkeypatch.setattr(model_runner_backend_module, "init_kv_cache", successful_init_cache)
    backend.initialize_kv_cache(config)
    assert backend.kv_cache_config is not None
    assert backend.kv_caches == ["cache-0", "cache-1"]
    assert backend.vllm_config.scheduler_config.max_num_seqs == 2


def test_initialize_rejects_config_for_different_registered_layers() -> None:
    backend = DiffusionKVModelRunnerBackend(
        vllm_config=SimpleNamespace(),
        od_config=SimpleNamespace(),
        device="cuda:1",
    )
    backend._kv_cache_layer_adapters = {
        "expected-layer": SimpleNamespace(spec=SimpleNamespace(non_causal=False), kv_cache=None)
    }
    config = SimpleNamespace(
        kv_cache_groups=[SimpleNamespace(layer_names=["other-layer"])],
    )

    with pytest.raises(ValueError, match="layer mismatch"):
        backend.initialize_kv_cache(config)


def test_valid_sequence_and_context_install_into_native_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)

    assert runner.install_diffusion_kv_metadata(_metadata()) is True

    assert block_tables.append_calls == [
        (0, ([1, 2, 3], [4, 5]), True),
        (1, ([6, 7], [8]), True),
    ]
    assert block_tables.apply_calls == 1
    assert runner.get_diffusion_kv_row("req-0", 0) == 0
    assert runner.get_diffusion_kv_row("req-0", 0, "text") == 1
    sequence_binding = runner._resolve_paged_attention_row("req-0", 0, None)
    context_binding = runner._resolve_paged_attention_row("req-0", None, "text")
    assert (sequence_binding.row_index, sequence_binding.max_seq_len) == (0, 9)
    assert (context_binding.row_index, context_binding.max_seq_len) == (1, 5)


def test_block_table_mutations_invalidate_prepared_attention_batches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _runner(monkeypatch)
    assert runner.paged_attention_adapter is not None
    runner.paged_attention_adapter.invalidate_prepared_batches = Mock()

    runner.install_diffusion_kv_metadata(_metadata())
    runner.paged_attention_adapter.invalidate_prepared_batches.assert_called_once_with()

    runner.remove_diffusion_kv_requests(["req-0"])
    assert runner.paged_attention_adapter.invalidate_prepared_batches.call_count == 2

    runner.refresh_block_table_layout()
    assert runner.paged_attention_adapter.invalidate_prepared_batches.call_count == 3


def test_cache_layer_registration_is_frozen_after_physical_initialization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner, _, _ = _runner(monkeypatch)

    with pytest.raises(RuntimeError, match="cannot be changed"):
        runner.register_kv_cache_layers({})


def test_request_scoped_context_uses_one_row_across_sequences(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    first_sequence = _metadata().sequences[0]
    metadata = replace(
        _metadata(),
        sequences=(
            first_sequence,
            replace(
                first_sequence,
                sequence_id=1,
                block_ids=([9, 10, 11], [12, 13]),
            ),
        ),
    )

    assert runner.install_diffusion_kv_metadata(metadata) is True

    assert block_tables.append_calls == [
        (0, ([1, 2, 3], [4, 5]), True),
        (1, ([9, 10, 11], [12, 13]), True),
        (2, ([6, 7], [8]), True),
    ]
    assert runner.get_diffusion_kv_row("req-0", 0, "text") == 2
    assert runner.get_diffusion_kv_row("req-0", 1, "text") == 2


def test_row_block_count_uses_native_spec_semantics(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    spec = runner.kv_cache_config.kv_cache_groups[0].kv_cache_spec
    spec.max_num_blocks_per_req = lambda _config, token_len: math.ceil(token_len / 8)

    metadata = _metadata(
        sequence_block_ids=([1, 2], [4, 5]),
        contexts=(),
    )

    assert runner.install_diffusion_kv_metadata(metadata) is True
    assert block_tables.append_calls == [(0, ([1, 2], [4, 5]), True)]


def test_repeated_native_block_ids_are_allowed(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    metadata = _metadata(
        sequence_block_ids=([1, 1, 3], [4, 4]),
        contexts=(),
    )

    assert runner.install_diffusion_kv_metadata(metadata) is True
    assert block_tables.append_calls == [(0, ([1, 1, 3], [4, 4]), True)]


def test_generation_is_idempotent_stale_safe_and_conflict_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    metadata = _metadata()
    runner.install_diffusion_kv_metadata(metadata)

    assert runner.install_diffusion_kv_metadata(metadata) is False
    assert runner.install_diffusion_kv_metadata(replace(metadata, allocation_generation=2)) is False
    assert block_tables.apply_calls == 1

    with pytest.raises(ValueError, match="Stale Diffusion KV allocation generation"):
        runner.install_diffusion_kv_metadata(metadata)
    with pytest.raises(ValueError, match="Conflicting Diffusion KV allocation snapshot"):
        runner.install_diffusion_kv_metadata(_metadata(generation=3, sequence_block_ids=([9, 2, 3], [4, 5])))
    assert block_tables.apply_calls == 1


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_metadata(sequence_block_ids=([1, 2, 3],)), "block groups"),
        (_metadata(sequence_block_ids=([1, 2], [4, 5])), "expected 3"),
        (_metadata(sequence_block_ids=([1, 2, 32], [4, 5])), "outside"),
        (
            _metadata(
                seq_len=17,
                sequence_block_ids=([1, 2, 3, 4, 5], [6, 7, 8]),
                contexts=(),
            ),
            "exceeds row capacity",
        ),
    ],
)
def test_invalid_group_count_block_count_range_and_row_capacity_are_atomic(
    monkeypatch: pytest.MonkeyPatch,
    metadata: DiffusionKVMetadata,
    message: str,
) -> None:
    runner, block_tables, _ = _runner(monkeypatch)

    with pytest.raises(ValueError, match=message):
        runner.install_diffusion_kv_metadata(metadata)

    assert block_tables.append_calls == []
    assert block_tables.apply_calls == 0
    assert runner._diffusion_kv_identity_to_row == {}
    assert runner._diffusion_kv_free_rows == [5, 4, 3, 2, 1, 0]


def test_invalid_later_context_does_not_stage_earlier_valid_row(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    bad_context = DiffusionKVContextMetadata(
        context_id="text",
        cache_role="cross_attention",
        num_tokens=5,
        block_ids=([6, 32], [8]),
    )

    with pytest.raises(ValueError, match="outside"):
        runner.install_diffusion_kv_metadata(_metadata(contexts=(bad_context,)))

    assert block_tables.append_calls == []
    assert all(not group_table.staged for group_table in block_tables.block_tables)


def test_native_append_failure_rolls_back_staged_state_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    block_tables.fail_on_append_call = 2

    with pytest.raises(RuntimeError, match="fake append failure"):
        runner.install_diffusion_kv_metadata(_metadata())

    assert np.count_nonzero(block_tables.num_blocks.np) == 0
    assert all(not group_table.staged for group_table in block_tables.block_tables)
    assert runner._diffusion_kv_identity_to_row == {}
    assert runner._diffusion_kv_free_rows == [5, 4, 3, 2, 1, 0]


def test_native_apply_failure_rolls_back_staged_state_and_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    block_tables.fail_apply = True

    with pytest.raises(RuntimeError, match="fake apply failure"):
        runner.install_diffusion_kv_metadata(_metadata())

    assert block_tables.apply_calls == 1
    assert np.count_nonzero(block_tables.num_blocks.np) == 0
    assert all(not group_table.staged for group_table in block_tables.block_tables)
    assert all(group_table.clear_calls == 1 for group_table in block_tables.block_tables)
    assert runner._diffusion_kv_identity_to_row == {}
    assert runner._diffusion_kv_free_rows == [5, 4, 3, 2, 1, 0]


def test_adapter_and_global_row_capacity_are_enforced(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch, max_rows_per_request=2, max_num_seqs=1)
    second_context = DiffusionKVContextMetadata(
        context_id="image",
        cache_role="cross_attention",
        num_tokens=1,
        block_ids=([9], [10]),
    )

    with pytest.raises(ValueError, match="adapter limit is 2"):
        runner.install_diffusion_kv_metadata(_metadata(contexts=(_metadata().contexts[0], second_context)))
    assert block_tables.append_calls == []

    runner.install_diffusion_kv_metadata(_metadata())
    append_count = len(block_tables.append_calls)
    with pytest.raises(ValueError, match="only 0 rows are free"):
        runner.install_diffusion_kv_metadata(_metadata("req-1"))
    assert len(block_tables.append_calls) == append_count


def test_empty_and_duplicate_sequence_metadata_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    empty = DiffusionKVMetadata(request_id="req-0", allocation_generation=0, sequences=())
    duplicate_sequence = _metadata().sequences[0]
    duplicate = DiffusionKVMetadata(
        request_id="req-0",
        allocation_generation=0,
        sequences=(duplicate_sequence, duplicate_sequence),
    )

    with pytest.raises(ValueError, match="at least one sequence"):
        runner.install_diffusion_kv_metadata(empty)
    with pytest.raises(ValueError, match="Duplicate diffusion KV row identity"):
        runner.install_diffusion_kv_metadata(duplicate)
    assert block_tables.append_calls == []


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (_metadata(generation=-1), "non-negative integer"),
        (_metadata(request_id=""), "non-empty string"),
        (_metadata(sequence_block_ids=([1, 2, True], [4, 5])), "non-integer block ID"),
    ],
)
def test_request_generation_and_block_id_types_are_validated(
    monkeypatch: pytest.MonkeyPatch,
    metadata: DiffusionKVMetadata,
    message: str,
) -> None:
    runner, block_tables, _ = _runner(monkeypatch)

    with pytest.raises(ValueError, match=message):
        runner.install_diffusion_kv_metadata(metadata)

    assert block_tables.append_calls == []
    assert runner._diffusion_kv_request_states == {}


def test_cleanup_is_idempotent_and_reuses_native_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch, max_rows_per_request=2, max_num_seqs=1)
    runner.install_diffusion_kv_metadata(_metadata())

    assert runner.remove_diffusion_kv_requests(["missing"]) == 0
    assert runner.remove_diffusion_kv_requests(["req-0", "req-0"]) == 2
    assert block_tables.append_calls[-2:] == [(0, ([], []), True), (1, ([], []), True)]
    assert block_tables.num_blocks.np[:, :2].tolist() == [[0, 0], [0, 0]]
    assert runner.remove_diffusion_kv_requests(["req-0"]) == 0
    with pytest.raises(KeyError, match="No native diffusion KV row"):
        runner.get_diffusion_kv_row("req-0", 0)

    runner.install_diffusion_kv_metadata(_metadata("req-1"))
    assert runner.get_diffusion_kv_row("req-1", 0) == 0
    assert runner.get_diffusion_kv_row("req-1", 0, "text") == 1


def test_cached_step_does_not_reinstall_metadata() -> None:
    runner = object.__new__(DiffusionModelRunner)
    runner.od_config = SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.PAGED_SCHEDULER)
    runner.pipeline = object()
    runner.install_diffusion_kv_metadata = Mock()
    runner._supports_step_mode = Mock(return_value=False)
    scheduler_output = DiffusionSchedulerOutput(
        step_id=1,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData(request_ids=["req-0"]),
        finished_req_ids=set(),
        num_running_reqs=1,
        num_waiting_reqs=0,
    )

    with pytest.raises(ValueError, match="does not support step execution"):
        runner.execute_stepwise(scheduler_output)
    runner.install_diffusion_kv_metadata.assert_not_called()


def test_dense_cleanup_and_wake_are_noops_but_paged_wake_refreshes(monkeypatch: pytest.MonkeyPatch) -> None:
    runner, block_tables, _ = _runner(monkeypatch)
    runner.refresh_block_table_layout()
    assert block_tables.layout_refreshes == 1

    dense_backend = DiffusionKVModelRunnerBackend(
        vllm_config=SimpleNamespace(),
        od_config=SimpleNamespace(diffusion_kv_mode=DiffusionKVCacheMode.DENSE_LEGACY),
        device="cuda:1",
    )
    dense_backend.block_tables = Mock()
    assert dense_backend.remove_diffusion_kv_requests(["req-0"]) == 0
    dense_backend.refresh_block_table_layout()
    dense_backend.block_tables.init_block_table_layout_tensors.assert_not_called()
    with pytest.raises(ValueError, match="Dense diffusion execution"):
        dense_backend.install_diffusion_kv_metadata(_metadata())
