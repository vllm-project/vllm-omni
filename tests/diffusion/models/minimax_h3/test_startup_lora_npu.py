# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""A5 validation of MiniMax H3 startup LoRA fusion and online quantization."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTArchConfig
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.startup_lora import MiniMaxH3StartupLoRA
from vllm_omni.lora.layers.base_linear import OmniBaseLinearLayerWithLoRA
from vllm_omni.platforms import current_omni_platform
from vllm_omni.platforms.npu import is_a5
from vllm_omni.quantization.mxfp8_config import DiffusionMXFP8Config

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


def _prepare(monkeypatch):
    if not current_omni_platform.is_npu():
        pytest.skip("requires an NPU")
    device = current_omni_platform.get_torch_device()
    if not is_a5(device):
        pytest.skip("requires an Ascend A5")
    current_omni_platform.set_device(device)
    device = torch.empty(0, device=device).device
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    config = VllmConfig()
    current_omni_platform.init_diffusion_worker_vllm_config(config)
    return device, config


@pytest.mark.parametrize("checkpoint_format", ["native", "diffusers"])
@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_startup_merge_npu_quantizes_original_qkv_once(monkeypatch, checkpoint_format):
    import torch_npu

    device, config = _prepare(monkeypatch)
    with set_current_vllm_config(config), torch.device(device):
        transformer = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
        torch.nn.Module.__init__(transformer)
        transformer.arch = MiniMaxH3DiTArchConfig(num_attention_heads=4, attention_head_dim=32)
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        layer = linear.QKVParallelLinear(
            128, 32, 4, bias=False, params_dtype=torch.bfloat16, quant_config=DiffusionMXFP8Config()
        )
        block.attn.qkv_proj = layer
        transformer.blocks = torch.nn.ModuleList([block])
        generator = torch.Generator(device="cpu").manual_seed(781)
        source = torch.randn(384, 128, generator=generator, device="cpu", dtype=torch.bfloat16)
        a = torch.randn(4, 128, generator=generator, device="cpu") * 0.02
        b = torch.randn(384, 4, generator=generator, device="cpu") * 0.02
        name = "transformer.blocks.0.attn.qkv_proj"
        adapter = LoRAModel(1, 4, {name: LoRALayerWeights(name, 4, 4, a, b)})
        fusion = MiniMaxH3StartupLoRA(adapter, 0.7)
        original_quant = torch_npu.npu_dynamic_mx_quant
        calls = []

        def count_quant(tensor, *args, **kwargs):
            assert tensor.device == device
            calls.append(tensor.detach().clone())
            return original_quant(tensor, *args, **kwargs)

        original_addmm, original_cpu = torch.addmm, torch.Tensor.cpu
        fusion_devices = []

        def observe_addmm(input, mat1, mat2, **kwargs):
            fusion_devices.append((input.device, mat1.device, mat2.device))
            return original_addmm(input, mat1, mat2, **kwargs)

        def reject_weight_download(tensor, *args, **kwargs):
            if tensor.device.type == "npu" and tensor.numel() > 1:
                raise AssertionError("startup fusion downloaded a device tensor")
            return original_cpu(tensor, *args, **kwargs)

        monkeypatch.setattr(torch_npu, "npu_dynamic_mx_quant", count_quant)
        with monkeypatch.context() as transfer_guard:
            transfer_guard.setattr(torch, "addmm", observe_addmm)
            transfer_guard.setattr(torch.Tensor, "cpu", reject_weight_download)
            if checkpoint_format == "diffusers":
                transformer._diffusers_weights = True
                transformer._serialized_weight_map = {}
                transformer._rope_theta = 10000.0
                transformer.rope = torch.nn.Module()
                transformer.rope.register_buffer("inv_freq", torch.empty(transformer.arch.rope_inv_freq_len))
                canonical_source = source.view(4, 3, 32, 128).permute(1, 0, 2, 3).reshape(384, 128)
                weights = [
                    (f"transformer_blocks.0.attn.to_{label}.weight", chunk)
                    for label, chunk in zip(("q", "k", "v"), canonical_source.chunk(3), strict=True)
                ]
            else:
                weights = [("blocks.0.attn.qkv_proj.weight", source)]
            transformer.load_weights(weights, startup_lora=fusion)
        assert fusion_devices and all(all(value == device for value in call) for call in fusion_devices)
        layer.quant_method.process_weights_after_loading(layer)
        fusion.validate()
        assert len(calls) == 1
        canonical = source.view(4, 3, 32, 128).permute(1, 0, 2, 3).reshape(384, 128)
        expected = torch.addmm(canonical.float(), b, a, alpha=0.7).bfloat16()
        torch.testing.assert_close(calls[0].cpu(), expected, rtol=0, atol=0)
        values, scales = original_quant(expected.to(device), dst_type=torch_npu.float8_e4m3fn)
        assert torch.equal(layer.weight.cpu().view(torch.uint8), values.t().contiguous().cpu().view(torch.uint8))
        expected_scales = scales.reshape(384, -1, 2).transpose(0, 1).contiguous()
        assert torch.equal(layer.weight_scale.cpu().view(torch.uint8), expected_scales.cpu().view(torch.uint8))
        assert block.attn.qkv_proj is layer

        def reject_runtime_lora(*args, **kwargs):
            raise AssertionError("runtime LoRA executed for a fixed startup policy")

        monkeypatch.setattr(OmniBaseLinearLayerWithLoRA, "apply", reject_runtime_lora)
        with torch.inference_mode():
            output = layer(torch.ones(2, 128, dtype=torch.bfloat16, device=device))[0]
        assert torch.isfinite(output).all()


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_startup_adaln_warmup_hits_before_first_npu_projection(monkeypatch, tmp_path):
    from tests.diffusion.models.minimax_h3.test_startup_lora import _adaln_warmup_pipeline, _warmup_plans
    from vllm_omni.inputs.data import OmniDiffusionSamplingParams

    device, config = _prepare(monkeypatch)
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path, device=device, vllm_config=config)
    transformer = pipeline.transformer
    projection = transformer.blocks[0].adaln_proj
    generator = torch.Generator().manual_seed(562)
    a, b = torch.randn(2, 4, generator=generator), torch.randn(72, 2, generator=generator)
    name = "transformer.blocks.0.adaln_proj.linear"
    adapter = LoRAModel(1, 2, {name: LoRALayerWeights(name, 2, 2, a, b)})
    fused = MiniMaxH3StartupLoRA(adapter, 0.7).merge(
        "blocks.0.adaln_proj.linear.weight", projection.linear.weight.detach().cpu()
    )
    with torch.no_grad():
        projection.linear.weight.copy_(fused.to(device))
    projections = [projection, transformer.final_layer.adaln_proj]
    references = []
    cache = transformer.adaln_cache
    cache.max_bytes = 0
    with torch.inference_mode():
        for plan in _warmup_plans(pipeline):
            embedding = transformer.time_embedder(plan.to(device))
            references.append([tuple(value.cpu() for value in owner(embedding)) for owner in projections])
    cache.max_bytes = 256 * 1024**2
    with torch.inference_mode():
        pipeline.warmup_startup_adaln_cache()
    misses = cache.misses
    assert misses == 8
    pipeline._prepare_adaln_adapter(OmniDiffusionSamplingParams(num_inference_steps=4))

    def reject_projection(*args, **kwargs):
        raise AssertionError("AdaLN Linear executed after fixed-schedule startup warmup")

    for owner in projections:
        monkeypatch.setattr(owner.linear, "forward", reject_projection)
    with torch.inference_mode():
        for plan, expected in zip(_warmup_plans(pipeline), references, strict=True):
            embedding = transformer.time_embedder(plan.to(device))
            cache.prepare(embedding)
            for owner, reference in zip(projections, expected, strict=True):
                for actual, value in zip(owner(embedding), reference, strict=True):
                    torch.testing.assert_close(actual.cpu(), value, rtol=0, atol=0)
    assert cache.misses == misses and cache.hits == 8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_startup_npu_fp32_fusion_preserves_target_dtype_and_precision_policy(monkeypatch, dtype):
    import torch_npu
    from torch_npu.npu.npu_config import _CubeMathType

    device, _ = _prepare(monkeypatch)
    generator = torch.Generator().manual_seed(791)
    a, b = torch.randn(64, 128, generator=generator), torch.randn(192, 64, generator=generator)
    source = torch.randn(192, 128, generator=generator).bfloat16()
    name = "transformer.blocks.0.adaln_proj.linear"
    adapter = LoRAModel(1, 64, {name: LoRALayerWeights(name, 64, 64, a, b)})
    fusion = MiniMaxH3StartupLoRA(adapter, 0.7)
    monkeypatch.setattr(fusion, "_NPU_ROWS_PER_TILE", 64)
    previous_hf32 = torch_npu.npu.matmul.allow_hf32
    previous_cube = torch_npu.npu.matmul.cube_math_type
    try:
        # An explicit reduced-precision serving policy must be suspended for the
        # fusion and restored before the loader invokes its quantization.
        torch_npu.npu.matmul.allow_hf32 = True
        if previous_cube is not None:
            torch_npu.npu.matmul.cube_math_type = _CubeMathType.USE_HF32
        actual = fusion.merge("blocks.0.adaln_proj.linear.weight", source, output_dtype=dtype, device=device)
        assert actual.device == device and actual.dtype == dtype
        assert torch_npu.npu.matmul.allow_hf32
        if previous_cube is not None:
            assert torch_npu.npu.matmul.cube_math_type == _CubeMathType.USE_HF32
        else:
            assert torch_npu.npu.matmul.cube_math_type is None
        reference = (source.double() + 0.7 * (b.double() @ a.double())).to(dtype)
        rtol = {torch.float16: 0.001, torch.bfloat16: 0.008, torch.float32: 2e-6}[dtype]
        torch.testing.assert_close(actual.cpu(), reference, rtol=rtol, atol=5e-6)
        fusion.validate()
    finally:
        if previous_cube is not None:
            torch_npu.npu.matmul.cube_math_type = previous_cube
        torch_npu.npu.matmul.allow_hf32 = previous_hf32


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_startup_npu_partial_packed_qkv_keeps_unadapted_slices(monkeypatch):
    from vllm.lora.lora_weights import PackedLoRALayerWeights

    device, _ = _prepare(monkeypatch)
    a, b = torch.full((4, 128), 0.125), torch.full((128, 4), 0.25)
    name = "transformer.blocks.0.attn.qkv_proj"
    packed = PackedLoRALayerWeights(name, 4, [4, 4, 4], [a, None, None], [b, None, None], scaling=[1.0, 1.0, 1.0])
    fusion = MiniMaxH3StartupLoRA(LoRAModel(1, 4, {name: packed}), 0.7)
    source = torch.ones(384, 128, dtype=torch.bfloat16)
    actual = fusion.merge("blocks.0.attn.qkv_proj.weight", source, output_dtype=torch.float32, device=device)
    expected = source.float()
    expected[:128] += 0.7 * (b @ a)
    assert actual.device == device and actual.dtype == torch.float32
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-6, atol=1e-7)
    fusion.validate()


@hardware_test(res={"npu": "A5"}, num_cards=1)
def test_startup_npu_overflow_rejects_weights_and_restores_precision_policy(monkeypatch):
    import torch_npu

    device, _ = _prepare(monkeypatch)
    a, b = torch.full((4, 128), 500.0), torch.full((192, 4), 500.0)
    name = "transformer.blocks.0.adaln_proj.linear"
    adapter = LoRAModel(1, 4, {name: LoRALayerWeights(name, 4, 4, a, b)})
    fusion = MiniMaxH3StartupLoRA(adapter, 1.0)
    before_hf32 = torch_npu.npu.matmul.allow_hf32
    before_cube = torch_npu.npu.matmul.cube_math_type
    with pytest.raises(ValueError, match="non-finite weights"):
        fusion.merge(
            "blocks.0.adaln_proj.linear.weight", torch.ones(192, 128), output_dtype=torch.float16, device=device
        )
    assert torch_npu.npu.matmul.allow_hf32 == before_hf32
    assert torch_npu.npu.matmul.cube_math_type == before_cube
    with pytest.raises(ValueError, match="targets were not loaded"):
        fusion.validate()
