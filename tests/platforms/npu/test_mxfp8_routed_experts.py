# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from tests.helpers.mark import hardware_test
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    pytest.mark.skipif(
        not current_omni_platform.is_npu(),
        reason="MXFP8 routed experts require an NPU",
    ),
]


@hardware_test(res={"npu": "A3"}, num_cards=1)
def test_load_serialized_mxfp8_routed_experts(monkeypatch: pytest.MonkeyPatch):
    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.config import (
        FusedMoEConfig,
        FusedMoEParallelConfig,
        RoutingMethodType,
    )
    from vllm.model_executor.layers.fused_moe.expert_map_manager import (
        ExpertMapManager,
    )
    from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts
    from vllm_ascend.quantization.method_adapters import AscendFusedMoEMethod
    from vllm_ascend.quantization.methods import w8a8_mxfp8
    from vllm_ascend.quantization.methods.w8a8_mxfp8 import (
        AscendW8A8MXFP8DynamicFusedMoEMethod,
    )

    from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

    quant_config = DiffusionMXFP8Config(is_checkpoint_mxfp8_serialized=True)
    vllm_config = SimpleNamespace(
        quant_config=quant_config,
        compilation_config=SimpleNamespace(mode=None),
        model_config=SimpleNamespace(enforce_eager=True),
    )
    ascend_config = SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False))
    monkeypatch.setattr(w8a8_mxfp8, "get_current_vllm_config", lambda: vllm_config)
    monkeypatch.setattr(w8a8_mxfp8, "get_ascend_config", lambda: ascend_config)

    parallel_config = FusedMoEParallelConfig.make_no_parallel()
    moe_config = FusedMoEConfig(
        num_experts=2,
        experts_per_token=1,
        hidden_dim=64,
        intermediate_size=64,
        num_local_experts=2,
        num_logical_experts=2,
        activation=MoEActivation.SILU,
        device="npu:0",
        routing_method=RoutingMethodType.Default,
        moe_parallel_config=parallel_config,
        in_dtype=torch.bfloat16,
    )
    expert_map_manager = ExpertMapManager(
        max_num_batched_tokens=8,
        top_k=1,
        global_num_experts=2,
        num_redundant_experts=0,
        num_expert_group=None,
        moe_parallel_config=parallel_config,
        placement_strategy="linear",
        enable_eplb=False,
    )

    torch.npu.set_device(0)
    with torch.device("npu:0"):
        layer = RoutedExperts(
            layer_name="layers.0.mlp.experts.routed_experts",
            params_dtype=torch.bfloat16,
            moe_config=moe_config,
            quant_config=quant_config,
            expert_map_manager=expert_map_manager,
        )

    assert isinstance(layer.quant_method, AscendFusedMoEMethod)
    assert isinstance(
        layer.quant_method.quant_method,
        AscendW8A8MXFP8DynamicFusedMoEMethod,
    )
    assert layer.w13_weight.device.type == "npu"

    def fp8_weight(value: float) -> torch.Tensor:
        return torch.full((64, 64), value).to(torch.float8_e4m3fn)

    checkpoint = {
        "gate_proj.weight": fp8_weight(1.0),
        "up_proj.weight": fp8_weight(2.0),
        "down_proj.weight": fp8_weight(3.0),
        "gate_proj.weight_scale": torch.full((64, 2), 4, dtype=torch.uint8),
        "up_proj.weight_scale": torch.full((64, 2), 5, dtype=torch.uint8),
        "down_proj.weight_scale": torch.full((64, 2), 6, dtype=torch.uint8),
    }
    mapping = {
        "gate_proj.weight": (layer.w13_weight, "w1"),
        "up_proj.weight": (layer.w13_weight, "w3"),
        "down_proj.weight": (layer.w2_weight, "w2"),
        "gate_proj.weight_scale": (layer.w13_weight_scale, "w1"),
        "up_proj.weight_scale": (layer.w13_weight_scale, "w3"),
        "down_proj.weight_scale": (layer.w2_weight_scale, "w2"),
    }
    for expert_id in range(2):
        for weight_name, loaded_weight in checkpoint.items():
            param, shard_id = mapping[weight_name]
            param.weight_loader(
                param=param,
                loaded_weight=loaded_weight,
                weight_name=f"experts.{expert_id}.{weight_name}",
                shard_id=shard_id,
                expert_id=expert_id,
            )

    torch.npu.synchronize()
    assert torch.all(layer.w13_weight[:, :64].float().cpu() == 1)
    assert torch.all(layer.w13_weight[:, 64:].float().cpu() == 2)
    assert torch.all(layer.w2_weight.float().cpu() == 3)
    assert torch.all(layer.w13_weight_scale[:, :64].cpu() == 4)
    assert torch.all(layer.w13_weight_scale[:, 64:].cpu() == 5)
    assert torch.all(layer.w2_weight_scale.cpu() == 6)

    layer.quant_method.process_weights_after_loading(layer)
    assert layer.w13_weight.shape == (2, 64, 128)
    assert layer.w2_weight.shape == (2, 64, 64)
    assert layer.w13_weight_scale.shape == (2, 1, 128, 2)
    assert layer.w2_weight_scale.shape == (2, 1, 64, 2)
