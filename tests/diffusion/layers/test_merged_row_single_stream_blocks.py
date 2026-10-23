# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU TP regressions for single-stream blocks that project ``cat([attn, mlp])``.

Under TP these blocks keep attention (local heads) and the MLP (local channels)
sharded and contract both with one ``MergedRowParallelLinear``. For each rank this
checks that the output projection sees exactly that rank's slice of the dense
``cat([attn, mlp])`` input, that nothing is all-gathered, and that the per-rank
partial outputs sum (identity all-reduce) to the dense projection output.

FLUX.2-klein has the same coverage in ``tests/diffusion/models/flux2/test_flux2_klein_local_tp.py``.
"""

from dataclasses import dataclass
from typing import Any

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import default_unquantized_gemm
from vllm.model_executor.model_loader.weight_utils import default_weight_loader

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import ForwardContext
from vllm_omni.diffusion.layers.linear import MergedRowParallelLinear
from vllm_omni.diffusion.models.flux import flux_transformer as flux
from vllm_omni.diffusion.models.flux2 import flux2_transformer as flux2
from vllm_omni.diffusion.models.longcat_image import longcat_image_transformer as longcat

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_HEAD_DIM = 8
_SEQ = 7
_TEXT = 3


class _CPUAttention(torch.nn.Module):
    def __init__(self, num_heads, head_size, softmax_scale, causal=False, **kwargs):
        super().__init__()
        self.num_heads = num_heads
        self.scale = softmax_scale

    def forward(self, query, key, value, metadata=None):
        return F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), scale=self.scale
        ).transpose(1, 2)


def _no_gather(*args, **kwargs):
    raise AssertionError("sharded single-stream TP must not all-gather")


@dataclass
class _Case:
    module: Any
    proj: str  # attribute path of the MergedRowParallelLinear on the built module

    def build(self, heads):
        raise NotImplementedError

    def run(self, block, heads):
        raise NotImplementedError


class _Flux1(_Case):
    def build(self, heads):
        return flux.FluxSingleTransformerBlock(
            dim=heads * _HEAD_DIM, num_attention_heads=heads, attention_head_dim=_HEAD_DIM
        )

    def run(self, block, heads):
        torch.manual_seed(5)
        dim = heads * _HEAD_DIM
        angles = torch.randn(_SEQ, _HEAD_DIM // 2)
        return block(
            torch.randn(2, _SEQ - _TEXT, dim),
            torch.randn(2, _TEXT, dim),
            torch.randn(2, dim),
            image_rotary_emb=(angles.cos(), angles.sin()),
        )


class _LongCat(_Case):
    def build(self, heads):
        return longcat.LongCatImageSingleTransformerBlock(
            parallel_config=DiffusionParallelConfig(),
            dim=heads * _HEAD_DIM,
            num_attention_heads=heads,
            attention_head_dim=_HEAD_DIM,
        )

    def run(self, block, heads):
        torch.manual_seed(5)
        dim = heads * _HEAD_DIM
        angles = torch.randn(_SEQ, _HEAD_DIM)
        return block(
            torch.randn(2, _SEQ - _TEXT, dim),
            torch.randn(2, _TEXT, dim),
            torch.randn(2, dim),
            image_rotary_emb=(angles.cos(), angles.sin()),
        )


class _Flux2(_Case):
    local_tp = True

    def build(self, heads):
        return flux2.Flux2ParallelSelfAttention(
            parallel_config=DiffusionParallelConfig(),
            query_dim=heads * _HEAD_DIM,
            heads=heads,
            dim_head=_HEAD_DIM,
            mlp_ratio=3,
            bias=True,
            local_tp=self.local_tp,
        )

    def run(self, block, heads):
        torch.manual_seed(5)
        angles = torch.randn(_SEQ, _HEAD_DIM // 2)
        return block(torch.randn(2, _SEQ, heads * _HEAD_DIM), image_rotary_emb=(angles.cos(), angles.sin()))


_CASES = {
    "flux1": _Flux1(flux, "proj_out"),
    "longcat_image": _LongCat(longcat, "proj_out"),
    "flux2": _Flux2(flux2, "to_out"),
}


def _set_tp(monkeypatch, case, tp, rank):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    monkeypatch.setattr(case.module, "get_tensor_model_parallel_world_size", lambda: tp)


def _native(module):
    for sub in module.modules():
        if hasattr(sub, "forward_native") and type(sub).__name__ in {"RMSNorm", "RotaryEmbedding"}:
            sub.forward = sub.forward_native
    return module


@pytest.fixture(autouse=True)
def _cpu(monkeypatch):
    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *a, **kw: default_unquantized_gemm)
    for case in _CASES.values():
        monkeypatch.setattr(case.module, "Attention", _CPUAttention)
        if hasattr(case.module, "get_forward_context"):
            monkeypatch.setattr(case.module, "get_forward_context", ForwardContext)


def _get(module, path):
    for part in path.split("."):
        module = getattr(module, part)
    return module


@pytest.mark.parametrize("case_name", list(_CASES))
@pytest.mark.parametrize("tp,heads", [(2, 4), (4, 8)])
@torch.inference_mode()
def test_rank_slices_and_partial_sums_match_dense(monkeypatch, case_name, tp, heads):
    case = _CASES[case_name]
    _set_tp(monkeypatch, case, 1, 0)
    torch.manual_seed(17)
    baseline = _native(case.build(heads))
    for name, param in baseline.named_parameters():
        if "norm_q" in name or "norm_k" in name:
            param.fill_(1)
        else:
            param.normal_(std=0.1)
    proj = _get(baseline, case.proj)
    captured = []
    hook = proj.register_forward_hook(lambda module, args, output: captured.append((args[0].clone(), output)))
    case.run(baseline, heads)
    hook.remove()
    dense_input, dense_output = captured[0]
    dense_output = dense_output[0] if isinstance(dense_output, tuple) else dense_output
    attn_width = heads * _HEAD_DIM
    dense_attn, dense_mlp = dense_input.split([attn_width, dense_input.shape[-1] - attn_width], dim=-1)
    weights = {name: param.detach().clone() for name, param in baseline.named_parameters()}

    partial_sum = torch.zeros_like(dense_output)
    for rank in range(tp):
        _set_tp(monkeypatch, case, tp, rank)
        block = _native(case.build(heads))
        local_proj = _get(block, case.proj)
        assert type(local_proj) is MergedRowParallelLinear and local_proj.tp_size == tp
        for name, param in block.named_parameters():
            getattr(param, "weight_loader", default_weight_loader)(param, weights[name])
        monkeypatch.setattr(case.module, "tensor_model_parallel_all_gather", _no_gather, raising=False)
        monkeypatch.setattr(linear, "tensor_model_parallel_all_gather", _no_gather)
        monkeypatch.setattr(linear, "tensor_model_parallel_all_reduce", lambda value: value)
        captured.clear()
        hook = local_proj.register_forward_hook(lambda module, args, output: captured.append((args[0].clone(), output)))
        case.run(block, heads)
        hook.remove()
        local_input, local_output = captured[0]
        expected_input = torch.cat([dense_attn.chunk(tp, dim=-1)[rank], dense_mlp.chunk(tp, dim=-1)[rank]], dim=-1)
        torch.testing.assert_close(local_input, expected_input, rtol=1e-4, atol=1e-5)
        partial_sum += local_output[0] if isinstance(local_output, tuple) else local_output
    torch.testing.assert_close(partial_sum, dense_output, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("case_name", ["flux1", "longcat_image"])
def test_tp1_keeps_unsharded_projection(monkeypatch, case_name):
    case = _CASES[case_name]
    _set_tp(monkeypatch, case, 1, 0)
    block = case.build(4)
    assert block.proj_mlp.tp_size == 1 and block.proj_out.tp_size == 1


def test_flux1_unsupported_quantization_keeps_replicated_projections(monkeypatch):
    monkeypatch.setattr(flux, "merged_row_parallel_supported", lambda quant_config, prefix="": False)
    _set_tp(monkeypatch, _CASES["flux1"], 4, 1)
    block = _CASES["flux1"].build(8)
    assert block.proj_mlp.tp_size == 1 and block.proj_out.tp_size == 1
    assert block.attn.gather_output


@pytest.mark.parametrize(
    "overrides,parallel_overrides,eligible",
    [
        ({}, {}, True),
        ({"dtype": torch.float32}, {}, False),
        ({}, {"tensor_parallel_size": 1}, False),
        ({}, {"ulysses_degree": 2}, False),
        ({"quantization_config": "fp8"}, {}, False),
        ({"lora_path": "local-adapter"}, {}, False),
        ({"enable_layerwise_offload": True}, {}, False),
    ],
)
def test_flux2_local_tp_eligibility(tmp_path, overrides, parallel_overrides, eligible):
    parallel = DiffusionParallelConfig(**({"tensor_parallel_size": 4} | parallel_overrides))
    config = OmniDiffusionConfig(
        **({"model": str(tmp_path), "dtype": torch.bfloat16} | overrides),
        parallel_config=parallel,
    )
    assert flux2._use_local_single_stream_tp(config) is eligible


@pytest.mark.parametrize("kwargs", [{"local_tp": False}, {"heads": 6}])
def test_flux2_ineligible_shapes_keep_original_projection(monkeypatch, kwargs):
    case = _Flux2(flux2, "to_out")
    case.local_tp = kwargs.get("local_tp", True)
    _set_tp(monkeypatch, case, 4, 0)
    attention = case.build(kwargs.get("heads", 8))
    assert not attention.local_tp
    assert type(attention.to_qkv_mlp_proj) is linear.ColumnParallelLinear
    assert type(attention.to_out) is linear.ColumnParallelLinear
    assert attention.query_num_heads == attention.heads
