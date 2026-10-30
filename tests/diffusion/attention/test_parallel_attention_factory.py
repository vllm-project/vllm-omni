# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from vllm_omni.diffusion.attention.parallel import factory
from vllm_omni.diffusion.distributed import a2a_permute

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def parallel_context(monkeypatch):
    parallel = SimpleNamespace(ulysses_degree=2, ring_degree=1, allgather_degree=1)
    config = SimpleNamespace(parallel_config=parallel)
    monkeypatch.setattr(factory, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(factory, "get_forward_context", lambda: SimpleNamespace(omni_diffusion_config=config))
    monkeypatch.setattr(factory, "get_sp_group", lambda: Mock(ulysses_world_size=2))
    monkeypatch.setattr(factory, "get_sequence_parallel_world_size", lambda: 2 * parallel.allgather_degree)
    return parallel


def test_default_does_not_build_optional_extension(monkeypatch, parallel_context):
    unavailable = Mock(side_effect=RuntimeError("NCCL headers are unavailable"))
    monkeypatch.setattr(a2a_permute, "ensure_a2a_permute_available", unavailable)

    strategy = factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert strategy._ulysses_a2a_permute is False
    unavailable.assert_not_called()


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("allgather", [1, 2])
def test_explicit_exchange_setting_is_forwarded(monkeypatch, parallel_context, enabled, allgather):
    parallel_context.ulysses_a2a_permute = enabled
    parallel_context.allgather_degree = allgather
    strategy = Mock()
    name = "UlyssesAllGatherKVParallelAttention" if allgather > 1 else "UlyssesParallelAttention"
    monkeypatch.setattr(factory, name, strategy)

    factory.build_parallel_attention_strategy(scatter_idx=2, gather_idx=1, use_sync=False)

    assert strategy.call_args.kwargs["ulysses_a2a_permute"] is enabled
