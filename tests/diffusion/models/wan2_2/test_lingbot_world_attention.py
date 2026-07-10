# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import math
import platform
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F
from torch import nn

_MODULE_PATH = Path(__file__).parents[4] / "vllm_omni/diffusion/models/wan2_2/lingbot_world_transformer.py"


def _install_macos_vllm_stubs() -> None:
    if platform.system() != "Darwin":
        return

    def ensure_module(name: str) -> ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = ModuleType(name)
            sys.modules[name] = module
        return module

    for name in (
        "vllm",
        "vllm.distributed",
        "vllm.model_executor",
        "vllm.model_executor.layers",
        "vllm.model_executor.layers.linear",
        "vllm.model_executor.utils",
        "vllm_omni",
        "vllm_omni.diffusion",
        "vllm_omni.diffusion.attention",
        "vllm_omni.diffusion.attention.layer",
        "vllm_omni.diffusion.layers",
        "vllm_omni.diffusion.layers.rope",
    ):
        ensure_module(name)

    distributed = sys.modules["vllm.distributed"]
    distributed.get_tensor_model_parallel_rank = lambda: 0
    distributed.get_tensor_model_parallel_world_size = lambda: 1
    distributed.tensor_model_parallel_all_reduce = lambda value: value

    def set_weight_attrs(weight: torch.Tensor, attrs: dict) -> None:
        for name, value in attrs.items():
            setattr(weight, name, value)

    sys.modules["vllm.model_executor.utils"].set_weight_attrs = set_weight_attrs

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


def _load_module():
    assert _MODULE_PATH.exists(), "LingBot attention module has not been implemented"
    _install_macos_vllm_stubs()
    spec = importlib.util.spec_from_file_location("_lingbot_world_attention_under_test", _MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
    for projection in (attention.q, attention.k, attention.v, attention.o):
        with torch.no_grad():
            projection.weight.copy_(torch.eye(2))
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
        cache.key.view(torch.uint8).clone(),
        cache.value.view(torch.uint8).clone(),
        (cache.end, cache.absolute_end, cache.last_start, cache.sink_end),
    )


def _assert_cache_unchanged(cache, snapshot) -> None:
    key, value, metadata = snapshot
    assert torch.equal(cache.key.view(torch.uint8), key)
    assert torch.equal(cache.value.view(torch.uint8), value)
    assert (cache.end, cache.absolute_end, cache.last_start, cache.sink_end) == metadata


def _make_cross_cache(module, *, dtype: torch.dtype = torch.float32, device: str = "cpu"):
    return module.LingBotAttentionCache(
        key=torch.randn(1, 2, 1, 2, dtype=dtype, device=device),
        value=torch.randn(1, 2, 1, 2, dtype=dtype, device=device),
        end=2,
        absolute_end=2,
        last_start=0,
    )


def test_self_attention_repeated_offset_overwrites_then_later_offset_appends() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=6).self_attention[0]

    attention(_tokens(1, 2), cache=cache, current_start=0)
    attention(_tokens(10, 20), cache=cache, current_start=0)

    assert cache.end == 2
    assert cache.absolute_end == 2
    torch.testing.assert_close(cache.key[0, : cache.end, 0, 0], torch.tensor([10.0, 20.0]))

    attention(_tokens(3, 4), cache=cache, current_start=2)

    assert cache.end == 4
    assert cache.absolute_end == 4
    torch.testing.assert_close(cache.key[0, : cache.end, 0, 0], torch.tensor([10.0, 20.0, 3.0, 4.0]))


def test_self_attention_is_chunk_causal_without_masking_inside_current_chunk() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=8).self_attention[0]
    attention_inputs = _record_inputs(attention.attn)

    attention(_tokens(1, 2), cache=cache, current_start=0)
    first_keys = attention_inputs[-1][1].detach().clone()
    attention(_tokens(3, 4), cache=cache, current_start=2)
    second_keys = attention_inputs[-1][1].detach().clone()

    assert attention.attn.causal is False
    torch.testing.assert_close(first_keys[0, :, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second_keys[0, :, 0, 0], torch.tensor([1.0, 2.0, 3.0, 4.0]))


def test_self_attention_retains_sink_and_latest_local_history_after_eviction() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=1)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    attention_inputs = _record_inputs(attention.attn)

    attention(_tokens(1, 2), cache=cache, current_start=0)
    attention(_tokens(3, 4), cache=cache, current_start=2)
    attention(_tokens(5, 6), cache=cache, current_start=4)

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
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    first = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    second = _allocate_single_layer(module, max_tokens=4).self_attention[0]

    attention(_tokens(1, 2), cache=first, current_start=0)

    assert first.end == 2
    assert second.end == 0
    assert torch.count_nonzero(second.key) == 0
    attention(_tokens(8), cache=second, current_start=0)
    torch.testing.assert_close(first.key[0, : first.end, 0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(second.key[0, : second.end, 0, 0], torch.tensor([8.0]))


@pytest.mark.parametrize("attention_name", ["LingBotSelfAttention", "LingBotCrossAttention"])
def test_attention_keeps_checkpoint_qkvo_and_norm_parameter_names(attention_name: str) -> None:
    module = _load_module()
    attention = getattr(module, attention_name)(dim=4, num_heads=2)

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
    self_attention = module.LingBotSelfAttention(dim=4, num_heads=2, sink_tokens=0)
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

    self_output = self_attention(hidden_states, cache=cache.self_attention[0], current_start=0)
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
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]
    cos = torch.ones(2, 1)
    sin = torch.zeros(2, 1)
    rotary_inputs = _record_inputs(attention.rotary_embedding)

    attention(_tokens(1, 2), cache=cache, current_start=0, rotary_emb=(cos, sin))

    assert len(rotary_inputs) == 2
    torch.testing.assert_close(rotary_inputs[0][1], cos)
    torch.testing.assert_close(rotary_inputs[1][2], sin)


def test_self_attention_slices_full_rotary_table_at_current_token_offset() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]
    cos = torch.arange(6, dtype=torch.float32).unsqueeze(1)
    sin = -cos
    rotary_inputs = _record_inputs(attention.rotary_embedding)

    attention(_tokens(1, 2), cache=cache, current_start=4, rotary_emb=(cos, sin))

    expected_cos = torch.tensor([[4.0], [5.0]])
    expected_sin = -expected_cos
    torch.testing.assert_close(rotary_inputs[0][1], expected_cos)
    torch.testing.assert_close(rotary_inputs[1][2], expected_sin)


def test_attention_block_owns_checkpoint_parent_namespaces() -> None:
    module = _load_module()
    block = module.LingBotAttentionBlock(dim=4, num_heads=2, sink_tokens=1, prefix="blocks.0")

    names = set(block.state_dict())
    children = ("q", "k", "v", "o", "norm_q", "norm_k")
    expected = {f"{parent}.{child}.weight" for parent in ("self_attn", "cross_attn") for child in children}

    assert expected <= names


@pytest.mark.parametrize(
    ("current_start", "message"),
    [
        (0, "precedes"),
        (3, "overlaps"),
        (5, "contiguous"),
    ],
    ids=("backward", "overlap", "gap"),
)
def test_rejected_new_chunk_offsets_preserve_self_cache(current_start: int, message: str) -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=6).self_attention[0]
    attention(_tokens(1, 2), cache=cache, current_start=0)
    attention(_tokens(3, 4), cache=cache, current_start=2)
    snapshot = _cache_snapshot(cache)

    with pytest.raises(ValueError, match=message):
        attention(_tokens(9), cache=cache, current_start=current_start)

    _assert_cache_unchanged(cache, snapshot)


def test_unequal_repeated_chunk_preserves_self_cache() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    attention(_tokens(1, 2), cache=cache, current_start=0)
    snapshot = _cache_snapshot(cache)

    with pytest.raises(ValueError, match="same-size"):
        attention(_tokens(3), cache=cache, current_start=0)

    _assert_cache_unchanged(cache, snapshot)


def test_chunk_over_capacity_preserves_self_cache() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=1).self_attention[0]
    snapshot = _cache_snapshot(cache)

    with pytest.raises(ValueError, match="full current chunk"):
        attention(_tokens(1, 2), cache=cache, current_start=0)

    _assert_cache_unchanged(cache, snapshot)


def test_sink_and_current_chunk_overflow_preserves_self_cache() -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=2)
    _set_identity_attention(attention)
    cache = _allocate_single_layer(module, max_tokens=4).self_attention[0]
    attention(_tokens(1, 2), cache=cache, current_start=0)
    snapshot = _cache_snapshot(cache)

    with pytest.raises(ValueError, match="sink tokens and the full current chunk"):
        attention(_tokens(3, 4, 5), cache=cache, current_start=2)

    _assert_cache_unchanged(cache, snapshot)


def _make_invalid_self_cache(module, case: str):
    cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]
    if case == "rank":
        cache.key = torch.zeros(1, 2, 2)
        cache.value = torch.zeros(1, 2, 2)
    elif case == "key_value_shape":
        cache.value = torch.zeros(1, 1, 1, 2)
    elif case == "batch":
        cache.key = torch.zeros(2, 2, 1, 2)
        cache.value = torch.zeros(2, 2, 1, 2)
    elif case == "heads":
        cache.key = torch.zeros(1, 2, 2, 2)
        cache.value = torch.zeros(1, 2, 2, 2)
    elif case == "head_dim":
        cache.key = torch.zeros(1, 2, 1, 3)
        cache.value = torch.zeros(1, 2, 1, 3)
    elif case == "dtype":
        cache.key = cache.key.double()
        cache.value = cache.value.double()
    elif case == "capacity":
        cache.key = torch.zeros(1, 0, 1, 2)
        cache.value = torch.zeros(1, 0, 1, 2)
    elif case == "negative_end":
        cache.end = -1
    elif case == "end_past_capacity":
        cache.end = 3
    elif case == "negative_sink_end":
        cache.sink_end = -1
    elif case == "sink_past_end":
        cache.sink_end = 1
    elif case == "sink_past_configured":
        cache.end = 2
        cache.absolute_end = 2
        cache.last_start = 0
        cache.sink_end = 2
    elif case == "negative_absolute_end":
        cache.absolute_end = -1
    elif case == "negative_last_start":
        cache.end = 1
        cache.absolute_end = 1
        cache.last_start = -1
    elif case == "uninitialized_nonempty":
        cache.end = 1
        cache.absolute_end = 1
    elif case == "latest_chunk_not_retained":
        cache.end = 1
        cache.absolute_end = 4
        cache.last_start = 2
    elif case == "physical_end_after_absolute_end":
        cache.end = 2
        cache.absolute_end = 1
        cache.last_start = 0
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)
    return cache


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "key_value_shape",
        "batch",
        "heads",
        "head_dim",
        "dtype",
        "capacity",
        "negative_end",
        "end_past_capacity",
        "negative_sink_end",
        "sink_past_end",
        "sink_past_configured",
        "negative_absolute_end",
        "negative_last_start",
        "uninitialized_nonempty",
        "latest_chunk_not_retained",
        "physical_end_after_absolute_end",
    ],
)
def test_invalid_self_cache_is_rejected_before_attention_dispatch(case: str) -> None:
    module = _load_module()
    attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=1)
    cache = _make_invalid_self_cache(module, case)
    snapshot = _cache_snapshot(cache)
    attention_calls = _record_inputs(attention.attn)

    with pytest.raises((ValueError, RuntimeError)):
        attention(_tokens(7), cache=cache, current_start=0)

    assert attention_calls == []
    _assert_cache_unchanged(cache, snapshot)


def _make_invalid_cross_cache(module, case: str):
    cache = _make_cross_cache(module)
    if case == "rank":
        cache.key = torch.zeros(1, 2, 2)
        cache.value = torch.zeros(1, 2, 2)
    elif case == "key_value_shape":
        cache.value = torch.zeros(1, 1, 1, 2)
    elif case == "empty":
        cache.end = 0
    elif case == "end_past_capacity":
        cache.end = 3
    elif case == "batch":
        cache.key = torch.zeros(2, 2, 1, 2)
        cache.value = torch.zeros(2, 2, 1, 2)
    elif case == "heads":
        cache.key = torch.zeros(1, 2, 2, 2)
        cache.value = torch.zeros(1, 2, 2, 2)
    elif case == "head_dim":
        cache.key = torch.zeros(1, 2, 1, 3)
        cache.value = torch.zeros(1, 2, 1, 3)
    elif case == "dtype":
        cache.key = cache.key.double()
        cache.value = cache.value.double()
    elif case == "absolute_end":
        cache.absolute_end = 1
    elif case == "last_start":
        cache.last_start = 1
    elif case == "missing_last_start":
        cache.last_start = None
    elif case == "sink_end":
        cache.sink_end = 1
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(case)
    return cache


@pytest.mark.parametrize(
    "case",
    [
        "rank",
        "key_value_shape",
        "empty",
        "end_past_capacity",
        "batch",
        "heads",
        "head_dim",
        "dtype",
        "absolute_end",
        "last_start",
        "missing_last_start",
        "sink_end",
    ],
)
def test_invalid_cross_cache_is_rejected_before_attention_dispatch(case: str) -> None:
    module = _load_module()
    attention = module.LingBotCrossAttention(dim=2, num_heads=1)
    cache = _make_invalid_cross_cache(module, case)
    snapshot = _cache_snapshot(cache)
    attention_calls = _record_inputs(attention.attn)

    with pytest.raises((ValueError, RuntimeError)):
        attention(_tokens(1), None, cache=cache)

    assert attention_calls == []
    _assert_cache_unchanged(cache, snapshot)


def test_cross_cache_device_mismatch_is_rejected_before_attention_dispatch() -> None:
    module = _load_module()
    attention = module.LingBotCrossAttention(dim=2, num_heads=1)
    cache = _make_cross_cache(module, device="meta")
    attention_calls = _record_inputs(attention.attn)

    with pytest.raises(ValueError, match="device"):
        attention(_tokens(1), None, cache=cache)

    assert attention_calls == []


def test_cached_tensors_are_detached_and_do_not_alias_projected_tensors() -> None:
    module = _load_module()
    self_attention = module.LingBotSelfAttention(dim=2, num_heads=1, sink_tokens=0)
    cross_attention = module.LingBotCrossAttention(dim=2, num_heads=1)
    self_cache = _allocate_single_layer(module, max_tokens=2).self_attention[0]
    hidden_states = _tokens(1, 2).requires_grad_()
    encoder_hidden_states = _tokens(3, 4).requires_grad_()
    self_projected_keys = _record_outputs(self_attention.k)
    self_projected_values = _record_outputs(self_attention.v)
    cross_projected_keys = _record_outputs(cross_attention.k)
    cross_projected_values = _record_outputs(cross_attention.v)

    self_attention(hidden_states, cache=self_cache, current_start=0)
    _, cross_cache = cross_attention(hidden_states, encoder_hidden_states, cache=None)

    for cached in (self_cache.key, self_cache.value, cross_cache.key, cross_cache.value):
        assert cached.requires_grad is False
        assert cached.grad_fn is None
    assert self_cache.key.untyped_storage().data_ptr() != self_projected_keys[0].untyped_storage().data_ptr()
    assert self_cache.value.untyped_storage().data_ptr() != self_projected_values[0].untyped_storage().data_ptr()
    assert self_cache.key.untyped_storage().data_ptr() != hidden_states.untyped_storage().data_ptr()
    assert self_cache.value.untyped_storage().data_ptr() != hidden_states.untyped_storage().data_ptr()
    assert cross_cache.key.untyped_storage().data_ptr() != cross_projected_keys[0].untyped_storage().data_ptr()
    assert cross_cache.value.untyped_storage().data_ptr() != cross_projected_values[0].untyped_storage().data_ptr()
    assert cross_cache.key.untyped_storage().data_ptr() != encoder_hidden_states.untyped_storage().data_ptr()


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
