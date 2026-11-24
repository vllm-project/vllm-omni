# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.attention.parallel import factory
from vllm_omni.diffusion.distributed import a2a_permute, ulysses_transport

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def parallel_context(monkeypatch):
    parallel = SimpleNamespace(ulysses_degree=2, ring_degree=1, allgather_degree=1)
    config = SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    monkeypatch.setattr(factory, "get_sp_group", lambda: Mock(ulysses_world_size=2))
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 2 * parallel.allgather_degree)
    # "auto" probes the group collectively; these tests have no process group.
    no_cuda = ulysses_transport.UlyssesTopology(2, cuda=False)
    monkeypatch.setattr(ulysses_transport, "probe_ulysses_topology", lambda group, world_size: no_cuda)
    return parallel


def test_default_does_not_build_optional_extension(monkeypatch, parallel_context):
    unavailable = Mock(side_effect=RuntimeError("NCCL headers are unavailable"))
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", unavailable)

    strategy = factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert strategy._ulysses_a2a_permute is False
    unavailable.assert_not_called()


@pytest.mark.parametrize("backend", ["nccl", "copy_engine", "flashinfer"])
@pytest.mark.parametrize("allgather", [1, 2])
def test_explicit_backend_is_forwarded(monkeypatch, parallel_context, backend, allgather):
    parallel_context.ulysses_a2a_backend = backend
    parallel_context.allgather_degree = allgather
    strategy = Mock()
    name = "UlyssesAllGatherKVParallelAttention" if allgather > 1 else "UlyssesParallelAttention"
    monkeypatch.setattr(factory, name, strategy)

    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    kwargs = strategy.call_args.kwargs
    if allgather > 1:
        assert kwargs["ulysses_a2a_backend"] == backend
    else:
        assert kwargs["transports"] == ulysses_transport.explicit_plan(backend, heads_per_rank=None)


@pytest.mark.parametrize("allgather", [1, 2])
def test_deprecated_permute_selects_copy_engine(monkeypatch, parallel_context, allgather):
    parallel_context.ulysses_a2a_permute = True
    parallel_context.allgather_degree = allgather
    strategy = Mock()
    name = "UlyssesAllGatherKVParallelAttention" if allgather > 1 else "UlyssesParallelAttention"
    monkeypatch.setattr(factory, name, strategy)

    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    kwargs = strategy.call_args.kwargs
    if allgather > 1:
        assert kwargs["ulysses_a2a_backend"] == "copy_engine"
    else:
        assert kwargs["transports"] == ulysses_transport.explicit_plan("copy_engine", heads_per_rank=None)


def test_auto_keeps_nccl_for_ulysses_allgather(monkeypatch, parallel_context):
    parallel_context.allgather_degree = 2
    strategy = Mock()
    monkeypatch.setattr(factory, "UlyssesAllGatherKVParallelAttention", strategy)

    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert strategy.call_args.kwargs["ulysses_a2a_backend"] == "nccl"


@pytest.mark.parametrize("chunks, expected", [(0, 2), (4, 4), (3, 0)])
def test_overlap_chunks_are_resolved_per_layer(monkeypatch, parallel_context, chunks, expected):
    parallel_context.ulysses_a2a_backend = "copy_engine_overlap"
    parallel_context.ulysses_overlap_chunks = chunks
    strategy = Mock()
    monkeypatch.setattr(factory, "UlyssesParallelAttention", strategy)

    # 8 heads over 2 ranks: 4 heads per rank, two heads per chunk by default.
    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False, num_heads=8)

    assert strategy.call_args.kwargs["transports"].overlap_chunks == expected
