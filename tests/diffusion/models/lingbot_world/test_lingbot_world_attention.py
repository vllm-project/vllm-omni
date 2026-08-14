# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

_MODULE_PATH = Path(__file__).parents[4] / "vllm_omni/diffusion/models/lingbot_world/transformer.py"
_MODULE_NAME = "_lingbot_world_attention_under_test"
_MISSING_MODULE = object()
_STUBBED_MODULE_NAMES = (
    "vllm",
    "vllm.distributed",
    "vllm.model_executor",
    "vllm.model_executor.layers",
    "vllm.model_executor.layers.conv",
    "vllm.model_executor.layers.linear",
    "vllm.model_executor.model_loader",
    "vllm.model_executor.model_loader.weight_utils",
    "vllm.model_executor.utils",
    "vllm_omni",
    "vllm_omni.diffusion",
    "vllm_omni.diffusion.attention",
    "vllm_omni.diffusion.attention.layer",
    "vllm_omni.diffusion.layers",
    "vllm_omni.diffusion.layers.norm",
    "vllm_omni.diffusion.layers.rope",
    "vllm_omni.experimental",
    "vllm_omni.experimental.ar_diffusion",
    "vllm_omni.experimental.ar_diffusion.kv_cache",
    "vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention",
)


@pytest.fixture(autouse=True)
def _inference_context():
    with torch.inference_mode():
        yield


def _install_vllm_stubs() -> None:
    for name in _STUBBED_MODULE_NAMES:
        sys.modules[name] = ModuleType(name)

    distributed = sys.modules["vllm.distributed"]
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    distributed.tensor_model_parallel_all_reduce = lambda value: value

    def set_weight_attrs(weight: torch.Tensor, attrs: dict) -> None:
        for name, value in attrs.items():
            setattr(weight, name, value)

    sys.modules["vllm.model_executor.utils"].set_weight_attrs = set_weight_attrs
    sys.modules["vllm.model_executor.model_loader.weight_utils"].default_weight_loader = lambda param, loaded_weight: (
        param.data.copy_(loaded_weight)
    )

    class _Linear(nn.Module):
        def __init__(
            self,
            input_size: int,
            output_size: int,
            *,
            bias: bool = True,
            return_bias: bool = False,
            **kwargs,
        ) -> None:
            super().__init__()
            del kwargs
            self.return_bias = return_bias
            self.weight = nn.Parameter(torch.empty(output_size, input_size))
            self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.calls = 0

        def forward(self, value: torch.Tensor):
            self.calls += 1
            output = F.linear(value, self.weight, self.bias)
            return (output, self.bias) if self.return_bias else output

    linear = sys.modules["vllm.model_executor.layers.linear"]
    linear.ColumnParallelLinear = _Linear
    linear.RowParallelLinear = _Linear

    class _QKVLinear(nn.Module):
        def __init__(
            self,
            hidden_size: int,
            head_size: int,
            total_num_heads: int,
            total_num_kv_heads: int | None = None,
            *,
            bias: bool = True,
            **kwargs,
        ) -> None:
            super().__init__()
            del kwargs
            self.num_heads = total_num_heads
            self.num_kv_heads = total_num_kv_heads or total_num_heads
            output_size = (self.num_heads + 2 * self.num_kv_heads) * head_size
            self.weight = nn.Parameter(torch.empty(output_size, hidden_size))
            self.bias = nn.Parameter(torch.empty(output_size)) if bias else None
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            if self.bias is not None:
                nn.init.zeros_(self.bias)
            self.calls = 0

            def load_shard(param: torch.Tensor, loaded_weight: torch.Tensor, shard_id: str) -> None:
                shard_index = {"q": 0, "k": 1, "v": 2}[shard_id]
                shard_size = param.shape[0] // 3
                param.data.narrow(0, shard_index * shard_size, shard_size).copy_(loaded_weight)

            self.weight.weight_loader = load_shard
            if self.bias is not None:
                self.bias.weight_loader = load_shard

        def forward(self, value: torch.Tensor):
            self.calls += 1
            return F.linear(value, self.weight, self.bias), None

    linear.QKVParallelLinear = _QKVLinear

    class _Conv3dLayer(nn.Conv3d):
        def __init__(self, in_channels: int, out_channels: int, kernel_size, stride, **kwargs) -> None:
            super().__init__(in_channels, out_channels, kernel_size=kernel_size, stride=stride, **kwargs)

    sys.modules["vllm.model_executor.layers.conv"].Conv3dLayer = _Conv3dLayer

    class _LayerNorm(nn.LayerNorm):
        def __init__(self, dim: int, eps: float = 1e-6, elementwise_affine: bool = True) -> None:
            super().__init__(dim, eps=eps, elementwise_affine=elementwise_affine)

    sys.modules["vllm_omni.diffusion.layers.norm"].LayerNorm = _LayerNorm

    class _Attention(nn.Module):
        def __init__(self, *args, softmax_scale: float, **kwargs) -> None:
            super().__init__()
            del args
            self.softmax_scale = softmax_scale
            self.causal = kwargs["causal"]
            self.calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata=None,
        ) -> torch.Tensor:
            del attn_metadata
            self.calls.append((query.detach().clone(), key.detach().clone(), value.detach().clone()))
            scores = torch.einsum("bqhd,bkhd->bhqk", query, key) * self.softmax_scale
            weights = scores.softmax(dim=-1)
            return torch.einsum("bhqk,bkhd->bqhd", weights, value)

    sys.modules["vllm_omni.diffusion.attention.layer"].Attention = _Attention

    class _RotaryEmbeddingWan(nn.Module):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__()
            del args, kwargs
            self.calls: list[tuple[torch.Tensor, torch.Tensor]] = []

        def forward(self, value: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
            self.calls.append((cos.detach().clone(), sin.detach().clone()))
            return value

    sys.modules["vllm_omni.diffusion.layers.rope"].RotaryEmbeddingWan = _RotaryEmbeddingWan

    paged_attention = sys.modules["vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention"]

    class _PagedLayerContext:
        pass

    class _PagedLayerInputs:
        pass

    paged_attention.ARDiffusionPagedLayerContext = _PagedLayerContext
    paged_attention.ARDiffusionPagedLayerInputs = _PagedLayerInputs
    paged_attention.ar_diffusion_paged_attention = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("CUDA paged attention was not expected in this CPU test")
    )
    paged_attention.paged_write_attn = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("paged path was not expected in this direct-cache test")
    )


def _load_module():
    assert _MODULE_PATH.exists(), "LingBot attention module has not been implemented"
    module_names = (*_STUBBED_MODULE_NAMES, _MODULE_NAME)
    previous = {name: sys.modules.get(name, _MISSING_MODULE) for name in module_names}
    try:
        _install_vllm_stubs()
        spec = importlib.util.spec_from_file_location(_MODULE_NAME, _MODULE_PATH)
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        for name, original in previous.items():
            if original is _MISSING_MODULE:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


def test_load_module_restores_process_module_state() -> None:
    module_names = (*_STUBBED_MODULE_NAMES, _MODULE_NAME)
    previous = {name: sys.modules.get(name, _MISSING_MODULE) for name in module_names}

    _load_module()

    for name, original in previous.items():
        assert sys.modules.get(name, _MISSING_MODULE) is original


def test_allocate_lingbot_cache_creates_request_local_layer_storage() -> None:
    module = _load_module()

    cache = module.allocate_lingbot_cache(
        batch_size=2,
        num_layers=3,
        max_tokens=7,
        num_local_heads=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert len(cache.self_attention) == 3
    assert len(cache.cross_attention) == 3
    assert cache.cross_attention == [None, None, None]
    assert cache.self_attention[0].key.shape == (2, 7, 2, 4)
    assert cache.self_attention[0].value.shape == (2, 7, 2, 4)
    assert cache.self_attention[0].key.dtype == torch.float32
    assert cache.self_attention[0].end == 0
    assert cache.self_attention[0].key.data_ptr() != cache.self_attention[1].key.data_ptr()


def _allocate_single_layer(module, *, max_tokens: int):
    return module.allocate_lingbot_cache(
        batch_size=1,
        num_layers=1,
        max_tokens=max_tokens,
        num_local_heads=1,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def _set_identity_attention(attention: nn.Module) -> None:
    qkv = getattr(attention, "qkv", None)
    if qkv is not None:
        projections = (qkv, attention.o)
    else:
        projections = (attention.q, attention.k, attention.v, attention.o)
    for projection in projections:
        with torch.no_grad():
            identity = torch.eye(2)
            projection.weight.copy_(torch.cat((identity, identity, identity)) if projection is qkv else identity)
            if projection.bias is not None:
                projection.bias.zero_()
    attention.norm_q = nn.Identity()
    attention.norm_k = nn.Identity()


def _tokens(*values: float) -> torch.Tensor:
    return torch.tensor([[[value, 0.0] for value in values]])


def _record_inputs(module: nn.Module) -> list[tuple]:
    records: list[tuple] = []
    module.register_forward_pre_hook(lambda _module, args: records.append(args))
    return records


def _record_outputs(module: nn.Module) -> list[torch.Tensor]:
    records: list[torch.Tensor] = []

    def record(_module, _args, output) -> None:
        records.append(output[0] if isinstance(output, tuple) else output)

    module.register_forward_hook(record)
    return records


def _cache_snapshot(cache) -> tuple[torch.Tensor, torch.Tensor, tuple[int, int, int | None, int]]:
    return (
        cache.key.detach().view(torch.uint8).clone(),
        cache.value.detach().view(torch.uint8).clone(),
        (cache.end, cache.absolute_end, cache.last_start, cache.sink_end),
    )


def _assert_cache_unchanged(cache, snapshot) -> None:
    key, value, metadata = snapshot
    assert torch.equal(cache.key.detach().view(torch.uint8), key)
    assert torch.equal(cache.value.detach().view(torch.uint8), value)
    assert (cache.end, cache.absolute_end, cache.last_start, cache.sink_end) == metadata


def test_self_attention_repeated_offset_overwrites_then_later_offset_appends() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=6).self_attention[0]

    attention(_tokens(1, 2), cache=cache, current_start=0, sink_tokens=0)
    attention(_tokens(10, 20), cache=cache, current_start=0, sink_tokens=0)

    assert cache.end == 2
    assert cache.absolute_end == 2
    torch.testing.assert_close(cache.key[0, : cache.end, 0, 0], torch.tensor([10.0, 20.0]))

    attention(_tokens(3, 4), cache=cache, current_start=2, sink_tokens=0)

    assert cache.end == 4
    assert cache.absolute_end == 4
    torch.testing.assert_close(cache.key[0, : cache.end, 0, 0], torch.tensor([10.0, 20.0, 3.0, 4.0]))


def test_self_attention_uses_one_fused_qkv_projection() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]

    attention(_tokens(1, 2), cache=cache, current_start=0, sink_tokens=0)

    assert attention.qkv.calls == 1


def test_self_attention_routes_paged_inputs_through_engine_owned_cache() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    paged_inputs = module.ARDiffusionPagedLayerInputs()
    calls = []

    def paged_write(inputs, query, key, value, k_act, v_act, scale):
        calls.append((inputs, query.clone(), key.clone(), value.clone(), k_act, v_act, scale))
        return query

    module.paged_write_attn = paged_write
    output = attention(
        _tokens(1, 2),
        cache=paged_inputs,
        current_start=0,
        sink_tokens=0,
        update_cache=False,
    )

    assert output.shape == (1, 2, 2)
    assert len(calls) == 1
    assert calls[0][0] is paged_inputs
    assert calls[0][4:6] == (None, None)
    assert attention.attn.calls == []
    assert not any(hasattr(attention, name) for name in ("q", "k", "v"))


def test_self_attention_is_chunk_causal_without_masking_inside_current_chunk() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=8).self_attention[0]
    attention_inputs = _record_inputs(attention.attn)

    attention(_tokens(1, 2), cache=cache, current_start=0, sink_tokens=0)
    first_keys = attention_inputs[-1][1].detach().clone()
    attention(_tokens(3, 4), cache=cache, current_start=2, sink_tokens=0)
    second_keys = attention_inputs[-1][1].detach().clone()

    assert attention.attn.causal is False
    torch.testing.assert_close(first_keys[0, :, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second_keys[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_self_attention_retains_sink_and_latest_local_history_after_eviction() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    attention_inputs = _record_inputs(attention.attn)

    attention(_tokens(1, 2), cache=cache, current_start=0, sink_tokens=1)
    attention(_tokens(3, 4), cache=cache, current_start=2, sink_tokens=1)
    attention(_tokens(5, 6), cache=cache, current_start=4, sink_tokens=1)

    visible_keys = attention_inputs[-1][1].detach().clone()
    assert cache.end == 4
    assert cache.absolute_end == 6
    torch.testing.assert_close(visible_keys[0, :, 0, 0], torch.tensor([1.0, 4.0, 5.0, 6.0]))
    torch.testing.assert_close(cache.key[0, : cache.end, 0, 0], torch.tensor([1.0, 4.0, 5.0, 6.0]))


def test_cross_attention_projects_encoder_kv_once_per_request() -> None:
    module = _load_module()
    attention = module.LingBotCrossAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    key_outputs = _record_outputs(attention.k)
    value_outputs = _record_outputs(attention.v)
    attention_inputs = _record_inputs(attention.attn)

    output, cache = attention(_tokens(1), _tokens(2, 3), cache=None)
    second_output, reused_cache = attention(_tokens(4), _tokens(20, 30), cache=cache)

    assert output.shape == second_output.shape == (1, 1, 2)
    assert reused_cache is cache
    assert len(key_outputs) == 1
    assert len(value_outputs) == 1
    assert cache.end == 2
    torch.testing.assert_close(cache.key[0, :, 0, 0], torch.tensor([2.0, 3.0]))
    torch.testing.assert_close(attention_inputs[-1][1], cache.key)


def test_self_attention_cache_is_isolated_between_requests() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    _set_identity_attention(attention)
    first = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    second = _allocate_single_layer(module, max_tokens=4).self_attention[0]

    attention(_tokens(1, 2), cache=first, current_start=0, sink_tokens=0)

    assert first.end == 2
    assert second.end == 0
    assert torch.count_nonzero(second.key) == 0
    attention(_tokens(8), cache=second, current_start=0, sink_tokens=0)
    torch.testing.assert_close(first.key[0, : first.end, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second.key[0, : second.end, 0, 0], torch.tensor([8.0]))


def test_cross_attention_keeps_checkpoint_qkvo_and_norm_parameter_names() -> None:
    module = _load_module()
    attention = module.LingBotCrossAttention(dim=4, num_heads=2)

    names = set(attention.state_dict())

    assert {
        "q.weight",
        "k.weight",
        "v.weight",
        "o.weight",
        "norm_q.weight",
        "norm_k.weight",
    } <= names
    assert callable(attention.norm_q.weight.weight_loader)


def test_tp_world_size_one_attention_output_shapes() -> None:
    module = _load_module()
    self_attention = module.LingBotSelfAttention(dim=4, num_heads=2)
    cross_attention = module.LingBotCrossAttention(dim=4, num_heads=2)
    cache = module.allocate_lingbot_cache(
        batch_size=2,
        num_layers=1,
        max_tokens=5,
        num_local_heads=2,
        head_dim=2,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    hidden_states = torch.randn(2, 3, 4)
    encoder_hidden_states = torch.randn(2, 5, 4)

    self_output = self_attention(hidden_states, cache=cache.self_attention[0], current_start=0, sink_tokens=0)
    cross_output, cache.cross_attention[0] = cross_attention(
        hidden_states,
        encoder_hidden_states,
        cache=cache.cross_attention[0],
    )

    assert self_output.shape == hidden_states.shape
    assert cross_output.shape == hidden_states.shape
    assert cache.cross_attention[0] is not None


def test_self_attention_applies_rotary_embedding_to_current_query_and_key() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1)
    cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]
    cos = torch.ones(2, 1)
    sin = torch.zeros(2, 1)
    rotary_inputs = _record_inputs(attention.rotary_embedding)

    attention(_tokens(1, 2), cache=cache, current_start=0, rotary_emb=(cos, sin), sink_tokens=0)

    assert len(rotary_inputs) == 2
    torch.testing.assert_close(rotary_inputs[0][1], cos)
    torch.testing.assert_close(rotary_inputs[1][2], sin)


def test_attention_block_owns_checkpoint_parent_namespaces() -> None:
    module = _load_module()
    block = module.LingBotAttentionBlock(dim=4, num_heads=2, prefix="blocks.0")

    names = set(block.state_dict())
    expected = {
        *(f"self_attn.{child}.weight" for child in ("qkv", "o", "norm_q", "norm_k")),
        *(f"cross_attn.{child}.weight" for child in ("q", "k", "v", "o", "norm_q", "norm_k")),
    }

    assert expected <= names


def test_cross_attention_caches_are_isolated_between_requests() -> None:
    module = _load_module()
    attention = module.LingBotCrossAttention(dim=2, num_heads=1)

    _, first = attention(_tokens(1), _tokens(2, 3), cache=None)
    _, second = attention(_tokens(1), _tokens(8, 9), cache=None)

    assert first.key.untyped_storage().data_ptr() != second.key.untyped_storage().data_ptr()
    assert first.value.untyped_storage().data_ptr() != second.value.untyped_storage().data_ptr()
    second_key = second.key.clone()
    first.key.zero_()
    torch.testing.assert_close(second.key, second_key)


def test_tp_rmsnorm_weight_loader_selects_rank_shard(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_module()
    monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 1)
    norm = module._LingBotRMSNorm(2, eps=1e-6)

    norm.weight.weight_loader(norm.weight, torch.tensor([10.0, 20.0, 30.0, 40.0]))

    torch.testing.assert_close(norm.weight, torch.tensor([30.0, 40.0]))
