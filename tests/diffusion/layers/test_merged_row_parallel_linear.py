# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for MergedRowParallelLinear's per-segment row-parallel layout.

Real vLLM parameter classes, weight loaders and LoRA wrappers run here; only the
TP rank/size metadata is substituted, and the all-reduce is replaced by an identity
so each rank's partial sum can be checked against the dense result.
"""

import pytest
import torch
import torch.nn.functional as F
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear
from vllm.model_executor.layers.utils import default_unquantized_gemm

from vllm_omni.diffusion.layers.linear import (
    MergedRowParallelLinear,
    merged_row_parallel_supported,
    to_rank_major,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

_SIZES = [16, 48]
_OUT = 12


@pytest.fixture(autouse=True)
def _cpu_dispatch(monkeypatch):
    from vllm.lora.layers import row_parallel_linear as lora_row

    monkeypatch.setattr(linear, "dispatch_unquantized_gemm", lambda *a, **kw: default_unquantized_gemm)
    for module in (linear, lora_row):
        monkeypatch.setattr(module, "tensor_model_parallel_all_reduce", lambda value: value)


def _set_tp(monkeypatch, tp, rank):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: tp)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)


def _layer(monkeypatch, tp, rank, **kwargs):
    _set_tp(monkeypatch, tp, rank)
    layer = MergedRowParallelLinear(_SIZES, _OUT, params_dtype=torch.float32, **kwargs)
    return layer


def _rank_cols(tensor, sizes, tp, rank):
    width = tensor.shape[1]
    seg = [size * width // sum(sizes) for size in sizes]
    return torch.cat([part.chunk(tp, dim=1)[rank] for part in tensor.split(seg, dim=1)], dim=1)


@pytest.mark.parametrize("tp,scale", [(1, 1), (2, 1), (4, 1), (8, 1), (2, 8), (4, 4)])
def test_rank_major_slices_each_segment(tp, scale):
    # scale > 1 models block/group scales: the on-disk width is a fraction of the
    # logical one, and segments must still map to whole elements per rank.
    width = sum(_SIZES) // scale
    tensor = torch.arange(3 * width, dtype=torch.float32).reshape(3, width)
    reordered = to_rank_major(tensor, 1, _SIZES, tp)
    for rank in range(tp):
        torch.testing.assert_close(
            reordered.chunk(tp, dim=1)[rank], _rank_cols(tensor, _SIZES, tp, rank), rtol=0, atol=0
        )
    assert sorted(reordered[0].tolist()) == tensor[0].tolist()


def test_rank_major_rejects_unaligned_segments():
    with pytest.raises(ValueError, match="cannot split an input dimension"):
        to_rank_major(torch.zeros(2, 16), 1, [4, 12], 8)


def test_constructor_rejects_indivisible_segment(monkeypatch):
    _set_tp(monkeypatch, 4, 0)
    with pytest.raises(ValueError, match="not divisible"):
        MergedRowParallelLinear([6, 10], 4)


@pytest.mark.parametrize("tp", [1, 2, 4])
@torch.inference_mode()
def test_checkpoint_load_and_partial_sums_match_dense(monkeypatch, tp):
    torch.manual_seed(3)
    weight = torch.randn(_OUT, sum(_SIZES))
    bias = torch.randn(_OUT)
    attn, mlp = torch.randn(2, 5, _SIZES[0]), torch.randn(2, 5, _SIZES[1])
    expected = F.linear(torch.cat([attn, mlp], dim=-1), weight, bias)

    partial_sum = torch.zeros_like(expected)
    for rank in range(tp):
        layer = _layer(monkeypatch, tp, rank)
        layer.weight.weight_loader(layer.weight, weight)
        layer.bias.weight_loader(layer.bias, bias)
        torch.testing.assert_close(layer.weight, _rank_cols(weight, _SIZES, tp, rank), rtol=0, atol=0)
        torch.testing.assert_close(layer.bias, bias, rtol=0, atol=0)
        local = torch.cat([attn.chunk(tp, dim=-1)[rank], mlp.chunk(tp, dim=-1)[rank]], dim=-1)
        output, output_bias = layer(local)
        assert output_bias is None
        partial_sum += output
    # Bias is added on rank 0 only, so the all-reduced sum carries it exactly once.
    torch.testing.assert_close(partial_sum, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("tp", [2, 4])
def test_v2_parameters_follow_their_input_dim(monkeypatch, tp):
    """Quantized-layout parameters: re-order along input_dim, replicate the rest."""
    for rank in range(tp):
        layer = _layer(monkeypatch, tp, rank)
        loader = layer.weight_loader_v2
        block = 4  # block-wise scales: one entry per 4 input columns
        weight = parameter.ModelWeightParameter(
            data=torch.empty(_OUT, sum(_SIZES) // tp), input_dim=1, output_dim=0, weight_loader=loader
        )
        block_scale = parameter.BlockQuantScaleParameter(
            data=torch.empty(_OUT // 4, sum(_SIZES) // block // tp), input_dim=1, output_dim=0, weight_loader=loader
        )
        per_tensor = parameter.PerTensorScaleParameter(data=torch.empty(1), weight_loader=loader)
        per_channel = parameter.ChannelQuantScaleParameter(
            data=torch.empty(_OUT, 1), output_dim=0, weight_loader=loader
        )

        full_weight = torch.arange(_OUT * sum(_SIZES), dtype=torch.float32).reshape(_OUT, -1)
        full_block = torch.arange(_OUT // 4 * sum(_SIZES) // block, dtype=torch.float32).reshape(_OUT // 4, -1)
        loader(weight, full_weight)
        loader(block_scale, full_block)
        loader(per_tensor, torch.tensor(0.5))
        loader(per_channel, torch.arange(_OUT, dtype=torch.float32).reshape(_OUT, 1))

        torch.testing.assert_close(weight.data, _rank_cols(full_weight, _SIZES, tp, rank), rtol=0, atol=0)
        torch.testing.assert_close(block_scale.data, _rank_cols(full_block, _SIZES, tp, rank), rtol=0, atol=0)
        assert per_tensor.data.item() == 0.5
        torch.testing.assert_close(per_channel.data.flatten(), torch.arange(_OUT, dtype=torch.float32))


def test_disable_tp_keeps_full_replicated_weight(monkeypatch):
    _set_tp(monkeypatch, 4, 3)
    layer = MergedRowParallelLinear(_SIZES, _OUT, params_dtype=torch.float32, disable_tp=True)
    weight = torch.randn(_OUT, sum(_SIZES))
    layer.weight.weight_loader(layer.weight, weight)
    assert layer.tp_size == 1
    torch.testing.assert_close(layer.weight, weight, rtol=0, atol=0)


def test_quantization_gate(monkeypatch):
    from vllm_omni.quantization import build_quant_config
    from vllm_omni.quantization.component_config import ComponentQuantizationConfig

    fp8 = build_quant_config("fp8")
    assert merged_row_parallel_supported(None)
    assert merged_row_parallel_supported(fp8)

    class _Positional:
        def get_name(self):
            return "gptq"

    assert not merged_row_parallel_supported(_Positional())
    routed = ComponentQuantizationConfig({"transformer.": _Positional()}, default_config=fp8)
    assert not merged_row_parallel_supported(routed, "transformer.blocks.0.proj_out")
    assert merged_row_parallel_supported(routed, "vae.proj_out")


def test_positional_input_params_are_rejected(monkeypatch):
    """GPTQ/AWQ act-order indices hold input positions; re-ordering would corrupt them."""

    class _GIdxMethod(linear.UnquantizedLinearMethod):
        def create_weights(self, layer, input_size_per_partition, *args, **kwargs):
            super().create_weights(layer, input_size_per_partition, *args, **kwargs)
            layer.register_parameter(
                "g_idx", torch.nn.Parameter(torch.zeros(input_size_per_partition, dtype=torch.int32), False)
            )

    monkeypatch.setattr(linear.LinearBase, "__init__", _patched_linear_base_init(_GIdxMethod))
    _set_tp(monkeypatch, 2, 0)
    with pytest.raises(ValueError, match="g_idx"):
        MergedRowParallelLinear(_SIZES, _OUT, params_dtype=torch.float32)


def _patched_linear_base_init(method_cls):
    original = linear.LinearBase.__init__

    def init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.quant_method = method_cls()

    return init


@pytest.mark.parametrize("tp", [1, 2, 4])
@torch.inference_mode()
def test_lora_a_follows_the_segment_layout(monkeypatch, tp):
    from vllm.lora.lora_model import LoRAModel
    from vllm.lora.lora_weights import LoRALayerWeights
    from vllm.lora.peft_helper import PEFTHelper
    from vllm.lora.request import LoRARequest

    from vllm_omni.diffusion.lora.layers import DiffusionMergedRowParallelLinearWithLoRA
    from vllm_omni.diffusion.lora.manager import DiffusionLoRAManager

    torch.manual_seed(11)
    weight = torch.randn(_OUT, sum(_SIZES))
    lora_a = torch.randn(2, sum(_SIZES))
    lora_b = torch.randn(_OUT, 2)
    attn, mlp = torch.randn(2, 3, _SIZES[0]), torch.randn(2, 3, _SIZES[1])
    full_input = torch.cat([attn, mlp], dim=-1)
    base_expected = F.linear(full_input, weight)
    delta = F.linear(F.linear(full_input, lora_a), lora_b)
    name = "transformer.proj_out"
    peft = PEFTHelper(r=2, lora_alpha=2, target_modules=["proj_out"])
    request = LoRARequest(lora_name="test", lora_int_id=5, lora_path="unused-by-memory-loader")

    partials = {0.5: torch.zeros_like(base_expected), None: torch.zeros_like(base_expected)}
    for rank in range(tp):
        layer = _layer(monkeypatch, tp, rank, bias=False)
        layer.weight.weight_loader(layer.weight, weight)
        pipeline = torch.nn.Module()
        pipeline.transformer = torch.nn.Module()
        pipeline.transformer.proj_out = layer
        manager = DiffusionLoRAManager(pipeline, device=torch.device("cpu"), dtype=torch.float32)
        adapter = LoRAModel(5, 2, {name: LoRALayerWeights(name, 2, 2, lora_a, lora_b)})
        monkeypatch.setattr(manager, "_load_adapter", lambda request: (adapter, peft))
        local = torch.cat([attn.chunk(tp, dim=-1)[rank], mlp.chunk(tp, dim=-1)[rank]], dim=-1)

        manager.set_active_adapter(request, lora_scale=0.5)
        wrapped = pipeline.transformer.proj_out
        assert isinstance(wrapped, DiffusionMergedRowParallelLinearWithLoRA)
        partials[0.5] += wrapped(local)[0]
        manager.set_active_adapter(None)
        partials[None] += wrapped(local)[0]

    torch.testing.assert_close(partials[0.5], base_expected + 0.5 * delta, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(partials[None], base_expected, rtol=1e-4, atol=1e-4)


def test_plain_row_parallel_lora_wrapper_does_not_claim_merged_layer(monkeypatch):
    from vllm.config.lora import LoRAConfig

    from vllm_omni.diffusion.lora.layers import (
        DiffusionMergedRowParallelLinearWithLoRA,
        DiffusionRowParallelLinearWithLoRA,
    )

    layer = _layer(monkeypatch, 2, 0)
    kwargs = dict(source_layer=layer, lora_config=LoRAConfig(), packed_modules_list=[])
    assert DiffusionMergedRowParallelLinearWithLoRA.can_replace_layer(**kwargs)
    assert not DiffusionRowParallelLinearWithLoRA.can_replace_layer(**kwargs)
