# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3 as cosmos3
from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3_multiview as multiview
from vllm_omni.diffusion.models.cosmos3.transformer_cosmos3_edge import Cosmos3EdgeVFMTransformer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def parallel_state(monkeypatch):
    """Keep TP linear construction and collectives, but run local GEMMs on CPU."""
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear

    def cpu_linear(method, layer, x, bias=None):
        return torch.nn.functional.linear(x, layer.weight, bias)

    monkeypatch.setattr(linear.UnquantizedLinearMethod, "apply", cpu_linear)

    def configure(size=1):
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: 0)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: size, raising=False)

    configure()
    return configure


class _Attention(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, hidden_states, **kwargs):
        return hidden_states * 0.125


def _gen_layer(dtype=torch.float32, transformer_cls=multiview.Cosmos3MultiviewVFMTransformer):
    layer = transformer_cls._gen_layer_cls(
        layer_idx=0,
        hidden_size=8,
        intermediate_size=24,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=4,
        rms_norm_eps=1e-6,
        mlp_cls=transformer_cls._gen_mlp_cls,
        cross_attention_cls=_Attention,
    ).to(dtype=dtype)
    with torch.no_grad():
        for parameter in layer.parameters():
            nn.init.uniform_(parameter, -0.25, 0.25)
        layer.post_attention_layernorm.weight.copy_(torch.linspace(0.5, 1.5, 8, dtype=dtype))
    layer._mlp_chunk_size = 8
    return layer


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("batch", [1, 2])
@pytest.mark.parametrize("sequence_length", [3, 8, 19])
@torch.inference_mode()
def test_multiview_gen_mlp_matches_full_sequence_with_bounded_chunks(dtype, batch, sequence_length):
    torch.manual_seed(42)
    layer = _gen_layer(dtype)
    packed = torch.randn(batch, sequence_length + 2, 8, dtype=dtype) * 20
    hidden = packed[:, 1:-1]  # Also exercise non-contiguous batched token slices.
    original = packed.clone()
    normalized = layer.post_attention_layernorm.forward_native(hidden)
    expected = layer.mlp(normalized)
    norm_outputs = []
    projection_rows = []

    def record_norm(module, args, output):
        assert args[0].numel() // args[0].shape[-1] <= layer._mlp_chunk_size
        norm_outputs.append(output.reshape(batch, -1, hidden.shape[-1]).clone())

    def record_projection(module, args, output):
        rows = args[0].numel() // args[0].shape[-1]
        assert rows <= layer._mlp_chunk_size
        assert output.numel() // output.shape[-1] == rows
        projection_rows.append(rows)

    layer.post_attention_layernorm.register_forward_hook(record_norm)
    for projection in layer.mlp.children():
        projection.register_forward_hook(record_projection)
    actual = layer._forward_mlp(hidden)

    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.cat(norm_outputs, dim=1), normalized, rtol=0, atol=0)
    torch.testing.assert_close(packed, original, rtol=0, atol=0)
    chunks = math.ceil(sequence_length / (layer._mlp_chunk_size // batch))
    assert len(norm_outputs) == chunks
    assert len(projection_rows) == chunks * len(list(layer.mlp.children()))
    assert actual.dtype == expected.dtype


@pytest.mark.parametrize("sequence_length", [3, 19])
@torch.inference_mode()
def test_multiview_gen_mlp_preserves_autocast_dtype(sequence_length):
    torch.manual_seed(42)
    layer = _gen_layer()
    hidden = torch.randn(1, sequence_length, 8)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        expected = layer.mlp(layer.post_attention_layernorm.forward_native(hidden))
        actual = layer._forward_mlp(hidden)
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sequence_length", [3, 19])
@torch.inference_mode()
def test_multiview_gen_mlp_reduces_once_before_residual(monkeypatch, parallel_state, dtype, sequence_length):
    torch.manual_seed(42)
    parallel_state(4)
    layers = [_gen_layer(dtype) for _ in range(4)]
    layer = layers[0]
    hidden = torch.randn(1, sequence_length, 8, dtype=dtype)
    residual = hidden + layer.input_layernorm(hidden) * 0.125
    normalized = layer.post_attention_layernorm.forward_native(residual)
    # Each layer supplies a distinct TP shard's contribution, using the same
    # normalized input. Simulate the collective without requiring four GPUs.
    contributions = [rank_layer.mlp(normalized) for rank_layer in layers]
    expected = residual + torch.stack(contributions).sum(dim=0)
    reductions = []

    def all_reduce(local_output):
        assert local_output.shape == hidden.shape
        torch.testing.assert_close(local_output, contributions[0])
        reductions.append(local_output.shape)
        return torch.stack([local_output, *contributions[1:]]).sum(dim=0)

    monkeypatch.setattr(multiview, "tensor_model_parallel_all_reduce", all_reduce)
    # Any reduction inside down_proj is a regression, including on short input.
    from vllm.model_executor.layers import linear

    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda _: pytest.fail("per-chunk TP reduction"))
    actual = layer(
        hidden,
        cached_kv=[(torch.zeros(1), torch.zeros(1))],
        freqs_gen=(torch.zeros(1), torch.zeros(1)),
    )

    torch.testing.assert_close(actual, expected)
    assert len(reductions) == 1
    assert layer.mlp.down_proj.reduce_results is False


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@torch.inference_mode()
def test_multiview_residuals_reuse_projection_outputs_without_mutating_input(dtype):
    torch.manual_seed(42)
    layer = _gen_layer(dtype)
    hidden = torch.randn(1, 19, 8, dtype=dtype)
    original = hidden.clone()
    residual = hidden + layer.input_layernorm(hidden) * 0.125
    expected = residual + layer._forward_mlp(residual)
    attention_pointers, mlp_pointers = [], []
    layer.cross_attention.register_forward_hook(
        lambda module, args, output: attention_pointers.append(output.data_ptr())
    )
    forward_mlp = layer._forward_mlp

    def record_mlp(hidden):
        assert hidden.data_ptr() == attention_pointers[-1]
        output = forward_mlp(hidden)
        mlp_pointers.append(output.data_ptr())
        return output

    layer._forward_mlp = record_mlp
    actual = layer(hidden, cached_kv=[(torch.zeros(1), torch.zeros(1))], freqs_gen=(torch.zeros(1), torch.zeros(1)))
    assert actual.data_ptr() == mlp_pointers[-1]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(hidden, original, rtol=0, atol=0)


def test_multiview_residuals_preserve_autograd_and_dtype_promotion():
    layer = _gen_layer()
    residual = torch.randn(1, 3, 8, requires_grad=True)
    projected = residual.sigmoid()  # Its original output is needed by backward.
    original = projected.detach().clone()
    output = layer._add_residual(projected, residual)
    output.sum().backward()
    torch.testing.assert_close(projected, original, rtol=0, atol=0)
    torch.testing.assert_close(residual.grad, 1 + original * (1 - original))
    with torch.inference_mode():
        projected = original.bfloat16()
        promoted = layer._add_residual(projected, residual.detach())
        assert promoted.dtype == torch.float32
        torch.testing.assert_close(promoted, residual.detach() + projected, rtol=0, atol=0)


@pytest.mark.parametrize("transformer_cls", [cosmos3.Cosmos3VFMTransformer, Cosmos3EdgeVFMTransformer])
@pytest.mark.parametrize("tp_size", [1, 4])
@torch.inference_mode()
def test_other_cosmos3_models_keep_full_sequence_mlp(monkeypatch, parallel_state, transformer_cls, tp_size):
    from vllm.model_executor.layers import linear

    torch.manual_seed(42)
    parallel_state(tp_size)
    layer = _gen_layer(transformer_cls=transformer_cls)
    assert type(layer) is cosmos3.Cosmos3GenDecoderLayer
    assert layer.mlp.down_proj.reduce_results is True
    assert transformer_cls._repeated_blocks == ["Cosmos3GenDecoderLayer"]
    hidden = torch.randn(1, 19, 8)
    normalized = layer.post_attention_layernorm.forward_native(hidden)
    # Keep the reference local; emulate a sum of identical TP contributions.
    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda x: x * tp_size)
    expected = layer.mlp(normalized)
    norm_shapes = []
    reductions = []
    layer.post_attention_layernorm.register_forward_pre_hook(lambda module, args: norm_shapes.append(args[0].shape))

    def all_reduce(local_output):
        reductions.append(local_output.shape)
        return local_output * tp_size

    monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", all_reduce)
    monkeypatch.setattr(multiview, "tensor_model_parallel_all_reduce", lambda _: pytest.fail("Multiview reduction"))
    actual = layer._forward_mlp(hidden)

    torch.testing.assert_close(actual, expected)
    assert norm_shapes == [hidden.shape]
    assert reductions == ([hidden.shape] if tp_size > 1 else [])


@pytest.mark.parametrize("dynamic", [False, True])
@torch.inference_mode()
def test_regional_compile_reuses_bounded_mlp_graphs(dynamic):
    from vllm_omni.diffusion.compile import regionally_compile

    torch.manual_seed(42)
    torch._dynamo.reset()
    layer = _gen_layer()
    eager_forward = layer.forward
    model = object.__new__(multiview.Cosmos3MultiviewVFMTransformer)
    nn.Module.__init__(model)
    model.gen_layers = nn.ModuleList([layer])
    mlp_graphs = []
    executed_rows = []

    def backend(graph, example_inputs):
        linears = [node for node in graph.graph.nodes if node.target == torch.nn.functional.linear]
        if not linears:
            return graph.forward
        # One chunk per graph, rather than an unrolled sequence of MLPs.
        assert len(linears) == 3
        mlp_graphs.append(graph)

        def run(*args):
            result = graph.forward(*args)
            output = result[0]
            rows = output.numel() // output.shape[-1]
            assert rows <= layer._mlp_chunk_size
            executed_rows.append(rows)
            return result

        return run

    try:
        regionally_compile(model, backend=backend, dynamic=dynamic)
        for length in (19, 35, 19):
            hidden = torch.randn(1, length, 8)
            kwargs = dict(
                k_und=torch.zeros(1),
                v_und=torch.zeros(1),
                freqs_cos=torch.zeros(1),
                freqs_sin=torch.zeros(1),
            )
            expected = eager_forward(hidden, **kwargs)
            actual = layer(hidden, **kwargs)
            torch.testing.assert_close(actual, expected)
        assert executed_rows == [8, 8, 3, 8, 8, 8, 8, 3, 8, 8, 3]
        assert 1 <= len(mlp_graphs) <= 2  # Full chunk and tail, shared across requests.
    finally:
        torch._dynamo.reset()
