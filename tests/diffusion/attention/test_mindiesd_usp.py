# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from dataclasses import dataclass, replace
from types import ModuleType
from typing import TypeAlias
from unittest.mock import Mock

import pytest
import torch

from vllm_omni.diffusion.attention.backends.abstract import AttentionMetadata
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.mindiesd_usp import (
    MindIESDUSPAdapter,
    MindIESDUSPOptions,
)
from vllm_omni.diffusion.data import DiffusionParallelConfig

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


@dataclass
class _ParallelConfigStub:
    enable_mindiesd_usp: bool = True
    ulysses_degree: int = 2
    ring_degree: int = 2
    allgather_degree: int = 1
    ulysses_mode: str = "strict"


@dataclass
class _SPGroupsStub:
    ulysses_group: object
    ring_group: object


USPErrorType: TypeAlias = type[BaseException]


def _parallel_config(**overrides):
    return replace(_ParallelConfigStub(), **overrides)


def _usp_module(usp_attention, usp_error: USPErrorType = RuntimeError) -> ModuleType:
    module = ModuleType("mindiesd.layers.usp")
    setattr(module, "usp_attention", usp_attention)
    setattr(module, "USPError", usp_error)
    return module


def _adapter(**overrides):
    config = _parallel_config(**overrides)
    groups = _SPGroupsStub(
        ulysses_group=object(),
        ring_group=object(),
    )
    return MindIESDUSPAdapter(MindIESDUSPOptions.from_parallel_config(config), groups)


def test_parallel_config_exposes_one_mindiesd_usp_switch():
    config = DiffusionParallelConfig(
        ulysses_degree=2,
        enable_mindiesd_usp=True,
    )

    assert config.sequence_parallel_size == 2
    assert config.enable_mindiesd_usp is True


def test_adapter_maps_vllm_owned_state_to_explicit_mindie_contract(monkeypatch):
    adapter = _adapter()
    usp_attention = Mock(return_value=torch.full((1, 3, 4, 8), 7.0))
    module = _usp_module(usp_attention)
    monkeypatch.setattr(adapter, "_load_usp_module", lambda: module)

    query = torch.randn(1, 3, 4, 8)
    key = torch.randn(1, 3, 4, 8)
    value = torch.randn(1, 3, 4, 8)
    metadata = AttentionMetadata()

    output = adapter.try_forward(
        query,
        key,
        value,
        attn_metadata=metadata,
        backend_name="FLASH_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
    )

    assert output is usp_attention.return_value
    usp_attention.assert_called_once_with(
        query,
        key,
        value,
        ulysses_group=adapter.sp_group.ulysses_group,
        kv_gather_group=adapter.sp_group.ring_group,
    )


def test_adapter_maps_pure_ring_to_kv_gather(monkeypatch):
    adapter = _adapter(ulysses_degree=1, ring_degree=2)
    usp_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    monkeypatch.setattr(
        adapter,
        "_load_usp_module",
        lambda: _usp_module(usp_attention),
    )
    query = torch.randn(1, 3, 4, 8)

    adapter.try_forward(
        query,
        query,
        query,
        attn_metadata=None,
        backend_name="FLASH_ATTN",
        causal=False,
        softmax_scale=8**-0.5,
        scatter_dim=2,
        gather_dim=1,
    )

    kwargs = usp_attention.call_args.kwargs
    assert kwargs["ulysses_group"] is None
    assert kwargs["kv_gather_group"] is adapter.sp_group.ring_group


def test_attention_delegates_before_native_sequence_parallel_collectives():
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    strategy = Mock()
    layer._get_active_parallel_strategy = Mock(return_value=strategy)
    layer._no_parallel_strategy = Mock()
    layer._active_paged_kv_adapter = Mock(return_value=None)
    layer._scheduler_paged_kv = False
    layer.paged_kv_cache_role = None
    layer._kv_cache_dtype = None
    layer._disable_kv_quant = False
    layer._kv_cache_skip_steps = None
    layer._kv_cache_skip_layers = None
    layer.attn_backend = Mock(get_name=Mock(return_value="FLASH_ATTN"))
    layer.causal = False
    layer.softmax_scale = 8**-0.5
    layer.scatter_idx = 2
    layer.gather_idx = 1
    expected = torch.zeros(1, 3, 4, 8)
    layer._mindiesd_usp_adapter = Mock(try_forward=Mock(return_value=expected))
    query = torch.randn(1, 3, 4, 8)

    output = layer._forward_impl(query, query, query)

    assert output is expected
    strategy.pre_attention.assert_not_called()
    strategy.post_attention.assert_not_called()


def test_attention_does_not_delegate_outside_sp_sharded_region():
    layer = Attention.__new__(Attention)
    torch.nn.Module.__init__(layer)
    layer._no_parallel_strategy = Mock()
    layer._get_active_parallel_strategy = Mock(return_value=layer._no_parallel_strategy)
    layer._active_paged_kv_adapter = Mock(return_value=None)
    layer._scheduler_paged_kv = False
    layer.paged_kv_cache_role = None
    layer._mindiesd_usp_adapter = Mock()
    layer.use_ring = False
    layer._with_kv_cache_dtype = Mock(side_effect=lambda metadata: metadata)
    layer._run_local_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    layer._no_parallel_strategy.pre_attention.return_value = (
        torch.zeros(1, 3, 4, 8),
        torch.zeros(1, 3, 4, 8),
        torch.zeros(1, 3, 4, 8),
        None,
        object(),
    )
    layer._no_parallel_strategy.post_attention.side_effect = lambda output, _ctx: output
    query = torch.randn(1, 3, 4, 8)

    layer._forward_impl(query, query, query)

    layer._mindiesd_usp_adapter.try_forward.assert_not_called()


@pytest.mark.parametrize(
    ("adapter_overrides", "call_overrides", "metadata"),
    [
        ({"enable_mindiesd_usp": False}, {}, None),
        ({}, {"backend_name": "TORCH_SDPA"}, None),
        ({}, {"causal": True}, None),
        ({}, {"softmax_scale": 0.25}, None),
        ({"ulysses_mode": "advanced_uaa"}, {}, None),
        ({"ulysses_degree": 1, "ring_degree": 1, "allgather_degree": 2}, {}, None),
        ({}, {}, AttentionMetadata(joint_query=torch.zeros(1, 1, 4, 8))),
        ({}, {}, AttentionMetadata(attn_mask=torch.ones(1, 3, dtype=torch.bool))),
        ({}, {}, AttentionMetadata(full_attn_spans=[[(0, 1)]])),
        ({}, {}, AttentionMetadata(extra={"kv_cache_dtype": "fp8"})),
    ],
)
def test_adapter_skips_semantics_not_covered_by_mindie(
    monkeypatch,
    adapter_overrides,
    call_overrides,
    metadata,
):
    adapter = _adapter(**adapter_overrides)
    usp_attention = Mock(return_value=torch.zeros(1, 3, 4, 8))
    monkeypatch.setattr(
        adapter,
        "_load_usp_module",
        lambda: _usp_module(usp_attention),
    )
    query = torch.randn(1, 3, 4, 8)
    call = {
        "attn_metadata": metadata,
        "backend_name": "FLASH_ATTN",
        "causal": False,
        "softmax_scale": 8**-0.5,
        "scatter_dim": 2,
        "gather_dim": 1,
    }
    call.update(call_overrides)

    assert adapter.try_forward(query, query, query, **call) is None
    usp_attention.assert_not_called()


def test_adapter_falls_back_only_for_structured_mindie_errors(monkeypatch):
    class USPError(RuntimeError):
        pass

    adapter = _adapter()
    usp_attention = Mock(side_effect=USPError("unsupported shape"))
    monkeypatch.setattr(
        adapter,
        "_load_usp_module",
        lambda: _usp_module(usp_attention, USPError),
    )
    query = torch.randn(1, 3, 4, 8)

    assert (
        adapter.try_forward(
            query,
            query,
            query,
            attn_metadata=None,
            backend_name="FLASH_ATTN",
            causal=False,
            softmax_scale=8**-0.5,
            scatter_dim=2,
            gather_dim=1,
        )
        is None
    )

    usp_attention.side_effect = ValueError("programming error")
    with pytest.raises(ValueError, match="programming error"):
        adapter.try_forward(
            query,
            query,
            query,
            attn_metadata=None,
            backend_name="FLASH_ATTN",
            causal=False,
            softmax_scale=8**-0.5,
            scatter_dim=2,
            gather_dim=1,
        )
