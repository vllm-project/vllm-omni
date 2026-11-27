# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import dataclasses
from typing import Any

import pytest
import torch
from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    MoEActivation,
    RoutingMethodType,
    UnquantizedFusedMoEMethod,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.modular_kernel import FusedMoEExpertsMonolithic
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.quantization.modelopt import (
    ModelOptFp8Config,
    ModelOptFp8MoEMethod,
    ModelOptNvFp4FusedMoE,
)
from vllm.model_executor.models.utils import PPMissingLayer

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    Cosmos3MixedPrecisionRuntime,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision import moe as moe_impl
from vllm_omni.diffusion.models.cosmos3.mixed_precision.moe import dequantize_fp8_experts

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def _fp8_moe_method(
    *,
    fp8_backend: Fp8MoeBackend = Fp8MoeBackend.TRITON,
    monolithic: bool = False,
    **parallel_overrides: object,
) -> ModelOptFp8MoEMethod:
    """Build the construction-time state of vLLM's ModelOpt FP8 MoE method."""
    method = object.__new__(ModelOptFp8MoEMethod)
    method.quant_config = object.__new__(ModelOptFp8Config)
    method.moe = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=4,
        intermediate_size=4,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.Custom,
        moe_parallel_config=dataclasses.replace(FusedMoEParallelConfig.make_no_parallel(), **parallel_overrides),
        in_dtype=torch.bfloat16,
    )
    method.moe_quant_config = None
    # The kernel is built only after weight loading, so install sees None.
    method.moe_kernel = None
    method.fp8_backend = fp8_backend
    method.experts_cls = FusedMoEExpertsMonolithic if monolithic else TritonExperts
    return method


class _RoutedExperts(torch.nn.Module):
    def __init__(self, quant_method: object) -> None:
        super().__init__()
        self.quant_method = quant_method


class _MoEBlock(torch.nn.Module):
    def __init__(self, quant_method: object) -> None:
        super().__init__()
        self.routed_experts = _RoutedExperts(quant_method)


def _fp8_expert_layer(*, flashinfer_order: bool) -> tuple[torch.nn.Module, torch.Tensor, torch.Tensor]:
    """Build processed FP8 expert weights and their BF16 [gate; up] and down references."""
    num_experts, hidden, intermediate = 4, 8, 4
    generator = torch.Generator().manual_seed(0)
    # Small integers and power-of-two scales keep FP8 and BF16 exact.
    w1, w3 = (torch.randint(-8, 9, (num_experts, intermediate, hidden), generator=generator).float() for _ in range(2))
    down = torch.randint(-8, 9, (num_experts, hidden, intermediate), generator=generator).float()
    scale = torch.tensor([1 / 64, 1 / 32, 1 / 16, 1 / 8])
    layer = torch.nn.Module()
    w13 = torch.cat([w3, w1] if flashinfer_order else [w1, w3], dim=1)
    layer.w13_weight = torch.nn.Parameter(w13.to(torch.float8_e4m3fn), requires_grad=False)
    layer.w2_weight = torch.nn.Parameter(down.to(torch.float8_e4m3fn), requires_grad=False)
    layer.w13_weight_scale = torch.nn.Parameter(scale.clone(), requires_grad=False)
    layer.w2_weight_scale = torch.nn.Parameter(scale.flip(0), requires_grad=False)
    gate_up = (torch.cat([w1, w3], dim=1) * scale.reshape(-1, 1, 1)).to(torch.bfloat16)
    return layer, gate_up, (down * scale.flip(0).reshape(-1, 1, 1)).to(torch.bfloat16)


def _transformer(
    gen_moes: list[torch.nn.Module],
    reasoner_moes: list[torch.nn.Module],
) -> torch.nn.Module:
    transformer = torch.nn.Module()
    transformer.gen_layers = torch.nn.ModuleList(gen_moes)
    transformer.language_model = torch.nn.Module()
    transformer.language_model.layers = torch.nn.ModuleList(reasoner_moes)
    return transformer


@pytest.mark.parametrize(
    ("reasoner", "paths"),
    [
        ("a16", ["generation", "generation", "reasoner"]),
        ("native", ["generation", "generation"]),
    ],
)
def test_install_discovers_generation_and_opt_in_reasoner_moe(reasoner: str, paths: list[str]) -> None:
    generation = [_MoEBlock(_fp8_moe_method()), _MoEBlock(_fp8_moe_method())]
    reasoner_block = _MoEBlock(_fp8_moe_method())
    unquantized_method = object.__new__(UnquantizedFusedMoEMethod)
    # The unquantized method is a CustomOp, so it must be a valid nn.Module.
    torch.nn.Module.__init__(unquantized_method)
    unquantized = _MoEBlock(unquantized_method)
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner=reasoner))

    runtime.install(_transformer([generation[0], unquantized, generation[1]], [reasoner_block]))

    assert runtime.has_routed_experts
    assert [record.path for record in runtime.routed_experts] == paths
    recorded_blocks = [*generation, reasoner_block][: len(paths)]
    assert [record.routed_experts for record in runtime.routed_experts] == [
        block.routed_experts for block in recorded_blocks
    ]


def test_install_skips_reasoner_moe_on_non_first_pipeline_stage() -> None:
    """Later PP stages hold only placeholders for the reasoner."""
    owned = _MoEBlock(_fp8_moe_method())
    transformer = _transformer([PPMissingLayer(), owned], [])
    transformer.language_model = PPMissingLayer()
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="a16"))

    runtime.install(transformer)

    assert [(record.routed_experts, record.path) for record in runtime.routed_experts] == [
        (owned.routed_experts, "generation")
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"fp8_backend": Fp8MoeBackend.MARLIN}, "backend", id="marlin"),
        pytest.param({"monolithic": True}, "monolithic", id="monolithic"),
        pytest.param({"ep_size": 2, "use_ep": True}, "expert parallel", id="ep"),
        pytest.param({"enable_eplb": True}, "EPLB", id="eplb"),
        pytest.param({"tp_size": 2}, "tensor/data parallel", id="tp"),
        pytest.param({"dp_size": 2}, "tensor/data parallel", id="dp"),
        pytest.param({"method": "nvfp4"}, "only ModelOpt FP8", id="non-fp8"),
    ],
)
def test_install_rejects_unsupported_moe(overrides: dict[str, Any], message: str) -> None:
    overrides = dict(overrides)
    if overrides.pop("method", None) == "nvfp4":
        method = object.__new__(ModelOptNvFp4FusedMoE)
    else:
        method = _fp8_moe_method(**overrides)
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig())

    with pytest.raises(ValueError, match=message):
        runtime.install(_transformer([_MoEBlock(method)], []))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        pytest.param({"lora_path": "/adapter"}, "LoRA", id="lora"),
        pytest.param({"cache_backend": "cache_dit"}, "Cache-DiT", id="cache-dit"),
    ],
)
def test_moe_maps_rejects_lora_and_cache_dit(overrides: dict[str, object], message: str) -> None:
    from vllm_omni.diffusion.models.cosmos3 import transformer_cosmos3

    od_config = OmniDiffusionConfig(model="test", **overrides)

    with pytest.raises(ValueError, match=message):
        transformer_cosmos3._validate_moe_mixed_precision_runtime(od_config)
    # Dense MAPS has no expert override to lose, so it still accepts the config.
    transformer_cosmos3._validate_mixed_precision_runtime(Cosmos3MixedPrecisionConfig(), od_config)


def test_generation_moe_dispatches_native_middle_and_a16_edges(monkeypatch) -> None:
    native, high = object(), object()
    native_calls, high_calls = [], []

    def _native_apply(*args):
        native_calls.append(args)
        return native

    def _apply_high_moe(*args):
        high_calls.append(args)
        return high

    monkeypatch.setattr(ModelOptFp8MoEMethod, "apply", _native_apply)
    monkeypatch.setattr(moe_impl, "apply_high_moe", _apply_high_moe)
    method = _fp8_moe_method()
    block = _MoEBlock(method)
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(first_steps=1, last_steps=1))
    runtime.install(_transformer([block], []))
    layer, x = torch.nn.Module(), torch.ones(1, 4, dtype=torch.bfloat16)
    topk_weights, topk_ids = torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32)

    # Two requests: the override must survive reset() without a second install.
    for _ in range(2):
        outputs = []
        for step in range(4):
            runtime.set_step(step, 4)
            outputs.append(method.apply(layer, x, topk_weights, topk_ids, None, None))
        assert outputs == [high, native, native, high]
        runtime.reset()

    assert native_calls == [(method, layer, x, topk_weights, topk_ids, None, None)] * 4
    assert high_calls == [(layer, x, topk_weights, topk_ids, Fp8MoeBackend.TRITON)] * 4
    assert block.routed_experts.quant_method is method


@dataclasses.dataclass
class _SharedExpertsOverlapKernel:
    """Stands in for a modular kernel that runs shared experts inside apply."""

    can_overlap_shared_experts: bool = True


def test_a16_moe_rejects_shared_experts_run_inside_the_kernel() -> None:
    method = _fp8_moe_method()
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig())
    runtime.install(_transformer([_MoEBlock(method)], []))
    method.moe_kernel = _SharedExpertsOverlapKernel()
    runtime.set_step(0, 4)
    x = torch.ones(1, 4, dtype=torch.bfloat16)

    with pytest.raises(NotImplementedError, match="shared experts"):
        method.apply(
            torch.nn.Module(), x, torch.ones(1, 1), torch.zeros(1, 1, dtype=torch.int32), torch.nn.Identity(), x
        )


def test_set_step_raises_when_moe_quant_method_is_replaced() -> None:
    block = _MoEBlock(_fp8_moe_method())
    runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig())
    runtime.install(_transformer([block], []))
    block.routed_experts.quant_method = _fp8_moe_method()

    with pytest.raises(RuntimeError, match="lost control"):
        runtime.set_step(0, 4)


@pytest.mark.parametrize(
    ("backend", "flashinfer_order"),
    [(Fp8MoeBackend.TRITON, False), (Fp8MoeBackend.FLASHINFER_CUTLASS, True)],
)
def test_fp8_expert_rebuild_matches_live_weights(backend: Fp8MoeBackend, flashinfer_order: bool) -> None:
    layer, reference_gate_up, reference_down = _fp8_expert_layer(flashinfer_order=flashinfer_order)

    w13, w2 = dequantize_fp8_experts(layer, backend)

    torch.testing.assert_close(w13, reference_gate_up, rtol=0, atol=0)
    torch.testing.assert_close(w2, reference_down, rtol=0, atol=0)
    # FSDP swaps in gathered weights between calls, so nothing may be cached.
    layer.w13_weight = torch.nn.Parameter(torch.zeros_like(layer.w13_weight), requires_grad=False)
    assert torch.count_nonzero(dequantize_fp8_experts(layer, backend)[0]) == 0
