# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""MAPS dispatch for real ModelOpt FP8 routed experts on vLLM's MoE kernels."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import vllm.forward_context as vllm_forward_context
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from vllm.config import DeviceConfig, LoRAConfig, VllmConfig, set_current_vllm_config
from vllm.distributed import cleanup_dist_env_and_memory, init_distributed_environment, initialize_model_parallel
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import FlashInferExperts
from vllm.model_executor.layers.fused_moe.oracle.fp8 import Fp8MoeBackend
from vllm.model_executor.layers.quantization.modelopt import ModelOptFp8Config, ModelOptFp8MoEMethod
from vllm.utils.torch_utils import set_default_torch_dtype
from vllm.v1.worker.workspace import init_workspace_manager, reset_workspace_manager

from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.layers.fused_moe import FusedMoE
from vllm_omni.diffusion.models.cosmos3.mixed_precision import (
    Cosmos3MixedPrecisionConfig,
    Cosmos3MixedPrecisionRuntime,
)
from vllm_omni.diffusion.models.cosmos3.mixed_precision.runtime import (
    Cosmos3MixedPrecisionLinearMethod,
)
from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_transformer import (
    HunyuanImage3Config,
    HunYuanSparseMoeBlock,
)
from vllm_omni.platforms import current_omni_platform

pytestmark = [
    pytest.mark.full_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": ["H100", "B200"]}, num_cards=1),
]

NUM_EXPERTS, HIDDEN, INTERMEDIATE, TOP_K, NUM_TOKENS = 8, 256, 128, 2, 32
# BF16 rounding of the intermediate and the output; the inputs keep outputs O(1).
HIGH_TOLERANCE = dict(rtol=2e-2, atol=2e-2)
HUNYUAN_FP8 = os.environ.get("HUNYUAN_IMAGE3_FP8_MODEL", "feizhai123/hunyuan-image3-modelopt-fp8")


@pytest.fixture
def moe_env(tmp_path):
    current_omni_platform.set_device(torch.device("cuda:0"))
    init_distributed_environment(
        world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{tmp_path / 'rendezvous'}"
    )
    config = VllmConfig(device_config=DeviceConfig(device="cuda"))
    try:
        with set_current_vllm_config(config):
            initialize_model_parallel(tensor_model_parallel_size=1)
            # The modular FusedMoE kernel draws scratch buffers from here.
            init_workspace_manager(torch.device("cuda"))
            yield config
    finally:
        reset_workspace_manager()
        cleanup_dist_env_and_memory()


def _reference_moe(
    x: torch.Tensor, topk_weights: torch.Tensor, topk_ids: torch.Tensor, w13: torch.Tensor, w2: torch.Tensor
) -> torch.Tensor:
    """FP32 per-expert SwiGLU loop over BF16 activations and dense [gate; up] weights."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    for expert in range(w13.shape[0]):
        token_idx, slot_idx = torch.where(topk_ids == expert)
        gate, up = w13[expert].float().chunk(2, dim=0)
        xe = x[token_idx].float()
        hidden = F.silu(xe @ gate.t()) * (xe @ up.t())
        out.index_add_(0, token_idx, (hidden @ w2[expert].float().t()) * topk_weights[token_idx, slot_idx, None])
    return out.to(x.dtype)


def _unpack_routing(
    hidden_states: torch.Tensor, gating_output: torch.Tensor, topk: int, renormalize: bool
) -> tuple[torch.Tensor, torch.Tensor]:
    """Custom router that reads routing packed as [weights | expert ids] by the test."""
    return gating_output[:, :topk].contiguous().float(), gating_output[:, topk:].contiguous().to(torch.int32)


def _load_fp8_routed_experts(
    experts: torch.nn.Module,
    w1: torch.Tensor,
    w3: torch.Tensor,
    w2: torch.Tensor,
    w13_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    w13_input_scale: torch.Tensor,
    w2_input_scale: torch.Tensor,
) -> None:
    for expert in range(w1.shape[0]):
        for shard, weight, weight_scale, input_scale in (
            ("w1", w1[expert], w13_scale[expert], w13_input_scale[expert]),
            ("w3", w3[expert], w13_scale[expert], w13_input_scale[expert]),
            ("w2", w2[expert], w2_scale[expert], w2_input_scale[expert]),
        ):
            prefix = "w2" if shard == "w2" else "w13"
            for name, value in (("weight", weight), ("weight_scale", weight_scale), ("input_scale", input_scale)):
                param = getattr(experts, f"{prefix}_{name}")
                param.weight_loader(param, value, f"experts.{expert}.{shard}.{name}", shard, expert)
    experts.quant_method.process_weights_after_loading(experts)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
@pytest.mark.parametrize("backend", ["triton", "flashinfer_cutlass", "triton_lora"])
def test_modelopt_fp8_moe_maps(moe_env, backend: str) -> None:
    if torch.cuda.get_device_capability(0) < (8, 9):
        pytest.skip("Requires native FP8 support")
    if backend == "flashinfer_cutlass" and not FlashInferExperts._supports_current_device():
        pytest.skip("FlashInfer CUTLASS fused MoE is not available on this device or install")
    moe_env.kernel_config.moe_backend = backend.removesuffix("_lora")
    if backend == "triton_lora":
        # Default serving of a MoE model with the PEFT LoRA backend (configure_diffusion_vllm_config).
        moe_env.lora_config = LoRAConfig(
            max_lora_rank=512, max_loras=1, max_cpu_loras=1, lora_dtype=torch.bfloat16, fully_sharded_loras=False
        )
    quant = ModelOptFp8Config.from_config({"quant_method": "modelopt", "quant_algo": "FP8"})

    def fp8_moe(prefix: str) -> torch.nn.Module:
        return FusedMoE(
            num_experts=NUM_EXPERTS,
            top_k=TOP_K,
            hidden_size=HIDDEN,
            intermediate_size=INTERMEDIATE,
            custom_routing_function=_unpack_routing,
            renormalize=False,
            quant_config=quant,
            params_dtype=torch.bfloat16,
            pcp_size=1,
            prefix=prefix,
        )

    with torch.device("cuda"), torch.inference_mode():
        plain = fp8_moe("plain.experts")
        maps = fp8_moe("gen_layers.0.experts")
        method = maps.routed_experts.quant_method
        assert type(method) is ModelOptFp8MoEMethod
        assert method.fp8_backend == Fp8MoeBackend[backend.removesuffix("_lora").upper()]
        assert method.moe.is_lora_enabled == (backend == "triton_lora")
        if backend == "triton_lora":
            assert method.experts_cls.supports_lora()

        transformer = torch.nn.Module()
        transformer.gen_layers = torch.nn.ModuleList([maps])
        runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="native"))
        runtime.install(transformer)

        # Small integers and power-of-two scales keep FP8 and BF16 exact.
        generator = torch.Generator("cuda").manual_seed(0)
        w1, w3 = (
            torch.randint(-8, 9, (NUM_EXPERTS, INTERMEDIATE, HIDDEN), generator=generator).float() for _ in range(2)
        )
        w2 = torch.randint(-8, 9, (NUM_EXPERTS, HIDDEN, INTERMEDIATE), generator=generator).float()
        w13_scale = 2.0 ** -torch.tensor([6, 7, 8, 6, 7, 8, 6, 7])
        w2_scale = 2.0 ** -torch.tensor([7, 6, 8, 8, 6, 7, 7, 6])
        input_scale = torch.full((NUM_EXPERTS,), 1 / 32)
        for layer in (plain, maps):
            _load_fp8_routed_experts(
                layer.routed_experts,
                w1.to(torch.float8_e4m3fn),
                w3.to(torch.float8_e4m3fn),
                w2.to(torch.float8_e4m3fn),
                w13_scale,
                w2_scale,
                input_scale,
                input_scale,
            )
        assert maps.routed_experts.quant_method is method

        x = torch.randn(NUM_TOKENS, HIDDEN, generator=generator).bfloat16()
        topk_ids = torch.stack([torch.randperm(NUM_EXPERTS, generator=generator)[:TOP_K] for _ in range(NUM_TOKENS)])
        topk_weights = torch.rand(NUM_TOKENS, TOP_K, generator=generator) + 0.25
        packed_routing = torch.cat([topk_weights, topk_ids.float()], dim=-1)
        dense_w13 = (torch.cat([w1, w3], dim=1) * w13_scale.reshape(-1, 1, 1)).bfloat16()
        dense_w2 = (w2 * w2_scale.reshape(-1, 1, 1)).bfloat16()
        high_reference = _reference_moe(x, topk_weights, topk_ids, dense_w13, dense_w2)

        with vllm_forward_context.set_forward_context(None, moe_env):
            native_baseline = plain(x, packed_routing)
            assert not torch.allclose(high_reference, native_baseline, **HIGH_TOLERANCE)
            for _ in range(2):
                for step in range(10):
                    runtime.set_step(step, 10)
                    actual = maps(x, packed_routing)
                    assert actual.shape == native_baseline.shape
                    if step < 3 or step >= 7:
                        torch.testing.assert_close(actual, high_reference, **HIGH_TOLERANCE)
                    else:
                        torch.testing.assert_close(actual, native_baseline, rtol=0, atol=0)
                runtime.reset()
        assert maps.routed_experts.quant_method is method


def _relative_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    return ((actual.float() - reference.float()).norm() / reference.float().norm()).item()


def _hunyuan_file(filename: str) -> str:
    local = Path(HUNYUAN_FP8) / filename
    return str(local) if local.is_file() else hf_hub_download(HUNYUAN_FP8, filename)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a CUDA GPU")
def test_hunyuan_image3_fp8_moe_maps(moe_env) -> None:
    """Real public FP8 MoE weights, with the shared expert inside the MoE runner."""
    if torch.cuda.get_device_capability(0) < (8, 9):
        pytest.skip("Requires native FP8 support")

    with open(_hunyuan_file("config.json")) as f:
        raw_config = json.load(f)
    config = HunyuanImage3Config(**raw_config)
    quant = ModelOptFp8Config.from_config(raw_config["quantization_config"])
    # Blackwell with FlashInfer would otherwise pick the monolithic TRT-LLM kernel, which MAPS rejects.
    moe_env.kernel_config.moe_backend = "triton"
    with open(_hunyuan_file("model.safetensors.index.json")) as f:
        weight_map = json.load(f)["weight_map"]
    prefix = "model.layers.0.mlp."
    checkpoint = {}
    for shard in sorted({file for name, file in weight_map.items() if name.startswith(prefix)}):
        with safe_open(_hunyuan_file(shard), framework="pt", device="cuda") as weights:
            checkpoint.update(
                {name[len(prefix) :]: weights.get_tensor(name) for name in weights.keys() if name.startswith(prefix)}
            )

    # Hunyuan stores fused gate/up projections as [up; gate]; every scale is per tensor.
    w3_fp8, w1_fp8 = torch.stack(
        [checkpoint[f"experts.{e}.gate_and_up_proj.weight"] for e in range(config.num_experts)]
    ).chunk(2, dim=1)
    w2_fp8 = torch.stack([checkpoint[f"experts.{e}.down_proj.weight"] for e in range(config.num_experts)])
    w13_scale = torch.stack(
        [checkpoint[f"experts.{e}.gate_and_up_proj.weight_scale"] for e in range(config.num_experts)]
    )
    w2_scale = torch.stack([checkpoint[f"experts.{e}.down_proj.weight_scale"] for e in range(config.num_experts)])
    w13_input_scale = torch.stack(
        [checkpoint[f"experts.{e}.gate_and_up_proj.input_scale"] for e in range(config.num_experts)]
    )
    w2_input_scale = torch.stack([checkpoint[f"experts.{e}.down_proj.input_scale"] for e in range(config.num_experts)])

    def load(block: torch.nn.Module) -> None:
        block.gate.weight.copy_(checkpoint["gate.wg.weight"])
        mlp = block.shared_mlp
        shared_up_fp8, shared_gate_fp8 = checkpoint["shared_mlp.gate_and_up_proj.weight"].chunk(2)
        for shard, weight in ((0, shared_gate_fp8), (1, shared_up_fp8)):
            shards = {
                "weight": weight,
                "weight_scale": checkpoint["shared_mlp.gate_and_up_proj.weight_scale"],
                "input_scale": checkpoint["shared_mlp.gate_and_up_proj.input_scale"],
            }
            for name, value in shards.items():
                param = getattr(mlp.gate_up_proj, name)
                param.weight_loader(param, value, shard)
        for name in ("weight", "weight_scale", "input_scale"):
            param = getattr(mlp.down_proj, name)
            param.weight_loader(param, checkpoint[f"shared_mlp.down_proj.{name}"])
        mlp.gate_up_proj.quant_method.process_weights_after_loading(mlp.gate_up_proj)
        mlp.down_proj.quant_method.process_weights_after_loading(mlp.down_proj)
        _load_fp8_routed_experts(
            block.experts.routed_experts,
            w1_fp8,
            w3_fp8,
            w2_fp8,
            w13_scale,
            w2_scale,
            w13_input_scale,
            w2_input_scale,
        )

    with torch.device("cuda"), torch.inference_mode():
        with set_default_torch_dtype(torch.bfloat16):
            plain = HunYuanSparseMoeBlock(config, quant, layer_id=0, prefix="model.layers.0.mlp")
            maps = HunYuanSparseMoeBlock(config, quant, layer_id=0, prefix="model.layers.1.mlp")
        method = maps.experts.routed_experts.quant_method
        assert type(method) is ModelOptFp8MoEMethod
        assert maps.experts.shared_experts is not None

        transformer = torch.nn.Module()
        transformer.gen_layers = torch.nn.ModuleList([maps])
        runtime = Cosmos3MixedPrecisionRuntime(Cosmos3MixedPrecisionConfig(reasoner="native"))
        runtime.install(transformer)
        assert len(runtime.routed_experts) == 1
        assert isinstance(maps.shared_mlp.gate_up_proj.quant_method, Cosmos3MixedPrecisionLinearMethod)
        assert isinstance(maps.shared_mlp.down_proj.quant_method, Cosmos3MixedPrecisionLinearMethod)
        load(plain)
        load(maps)
        assert maps.experts.routed_experts.quant_method is method

        def dense(name: str) -> torch.Tensor:
            return (checkpoint[f"{name}.weight"].float() * checkpoint[f"{name}.weight_scale"].float()).bfloat16()

        generator = torch.Generator("cuda").manual_seed(0)
        # Scale the inputs to the calibrated FP8 activation range.
        amax = 448 * checkpoint["shared_mlp.gate_and_up_proj.input_scale"].float()
        x = (torch.randn(64, config.hidden_size, generator=generator) * amax / 4).bfloat16()
        router_logits, _ = maps.gate(x)
        topk_weights, topk_ids = maps.experts.router.select_experts(x, router_logits)
        dense_w13 = (torch.cat([w1_fp8, w3_fp8], dim=1).float() * w13_scale.float().reshape(-1, 1, 1)).bfloat16()
        dense_w2 = (w2_fp8.float() * w2_scale.float().reshape(-1, 1, 1)).bfloat16()
        shared_up, shared_gate = dense("shared_mlp.gate_and_up_proj").float().chunk(2)
        shared = (F.silu(x.float() @ shared_gate.t()) * (x.float() @ shared_up.t())) @ dense(
            "shared_mlp.down_proj"
        ).float().t()
        routed = _reference_moe(x, topk_weights, topk_ids.long(), dense_w13, dense_w2)
        high_reference = (shared + routed.float()).bfloat16()

        with vllm_forward_context.set_forward_context(None, moe_env):
            native_baseline = plain(x)
            # The check below can tell A16 from native, and from A16 without the shared expert.
            assert _relative_error(native_baseline, high_reference) > 0.1
            assert _relative_error(routed, high_reference) > 0.1
            for step in range(10):
                runtime.set_step(step, 10)
                actual = maps(x)
                if step < 3 or step >= 7:
                    assert _relative_error(actual, high_reference) < 1e-2
                else:
                    torch.testing.assert_close(actual, native_baseline, rtol=0, atol=0)
            runtime.reset()
