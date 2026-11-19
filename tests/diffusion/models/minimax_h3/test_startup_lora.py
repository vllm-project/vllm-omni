# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Original-checkpoint LoRA fusion before TP/online quantization."""

import pytest
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.lora.lora_model import LoRAModel
from vllm.lora.lora_weights import LoRALayerWeights
from vllm.model_executor import parameter
from vllm.model_executor.layers import linear

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3DiTArchConfig
from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3DiTModel
from vllm_omni.diffusion.models.minimax_h3.pipeline_minimax_h3 import MiniMaxH3Pipeline
from vllm_omni.diffusion.models.minimax_h3.startup_lora import MiniMaxH3StartupLoRA
from vllm_omni.diffusion.sched.sigma_schedule import DMD2SigmaSchedule
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.lora.request import LoRARequest

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def cpu_adapter_loading_does_not_allocate_accelerator_pinned_memory(monkeypatch):
    import vllm.lora.lora_model as lora_model

    monkeypatch.setattr(lora_model, "PIN_MEMORY", False)


def _adapter(name="transformer.blocks.0.attn.qkv_proj", rows=6):
    generator = torch.Generator().manual_seed(281)
    a = torch.randn(2, 4, generator=generator)
    b = torch.randn(rows, 2, generator=generator)
    weights = LoRALayerWeights(name, 2, 3, a, b)
    return LoRAModel(1, 2, {name: weights}), a, b


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_startup_fuses_original_weights_with_fp32_accumulation(dtype, monkeypatch):
    model, a, b = _adapter()
    fusion = MiniMaxH3StartupLoRA(model, 0.7)
    monkeypatch.setattr(fusion, "_ROWS_PER_TILE", 1)
    original = torch.linspace(-1, 1, 24).reshape(6, 4).to(dtype)
    before = original.clone()
    actual = fusion.merge("blocks.0.attn.qkv_proj.weight", original)
    expected = torch.addmm(original.float(), b, a, alpha=1.5 * 0.7).to(dtype)
    tolerance = {"rtol": 1e-6, "atol": 5e-7} if dtype == torch.float32 else {"rtol": 0, "atol": 0}
    torch.testing.assert_close(actual, expected, **tolerance)
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    fusion.validate()


@pytest.mark.parametrize("size,rank", [(1, 0), (2, 0), (2, 1)])
@pytest.mark.parametrize("checkpoint_format", ["native", "diffusers"])
@pytest.mark.parametrize("lazy_materialize", [False, True])
def test_qkv_merges_after_layout_conversion_before_tp_loader(
    monkeypatch, size, rank, checkpoint_format, lazy_materialize
):
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: size)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: rank)
    with set_current_vllm_config(VllmConfig()):
        transformer = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
        torch.nn.Module.__init__(transformer)
        transformer.arch = MiniMaxH3DiTArchConfig(num_attention_heads=2, attention_head_dim=1)
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        layer = linear.QKVParallelLinear(4, 1, 2, bias=False, params_dtype=torch.bfloat16)
        block.attn.qkv_proj = layer
        transformer.blocks = torch.nn.ModuleList([block])
        adapter, a, b = _adapter()
        fusion = MiniMaxH3StartupLoRA(adapter, 0.7)
        source = torch.arange(24).reshape(6, 4).to(torch.bfloat16) / 16
        canonical = source[[0, 3, 1, 4, 2, 5]]
        expected = torch.addmm(canonical.float(), b, a, alpha=1.05).bfloat16()
        original_loader = layer.weight.weight_loader
        materialized_weight = layer.weight
        if lazy_materialize:
            # Mirror the online loader replacing its meta parameter on the first
            # split QKV chunk, while the model's params dictionary keeps the old one.
            layer._load_device = torch.device("cpu")
            layer.register_parameter(
                "weight", torch.nn.Parameter(torch.empty_like(layer.weight, device="meta"), requires_grad=False)
            )
        observed = []
        merge_devices = []
        original_merge = fusion.merge

        def observe_merge(*args, **kwargs):
            merge_devices.append(kwargs["device"])
            return original_merge(*args, **kwargs)

        def observe_before_quant(param, tensor, *args):
            observed.append(tensor.clone())
            if layer.weight.device.type == "meta":
                layer.register_parameter("weight", materialized_weight)
                del layer._load_device
            return original_loader(layer.weight, tensor, *args)

        monkeypatch.setattr(fusion, "merge", observe_merge)
        monkeypatch.setattr(layer.weight, "weight_loader", observe_before_quant, raising=False)
        if checkpoint_format == "diffusers":
            transformer._diffusers_weights = True
            transformer._serialized_weight_map = {}
            transformer._rope_theta = 10000.0
            transformer.rope = torch.nn.Module()
            transformer.rope.register_buffer("inv_freq", torch.empty(transformer.arch.rope_inv_freq_len))
            weights = [
                (f"transformer_blocks.0.attn.to_{label}.weight", value)
                for label, value in zip(("q", "k", "v"), canonical.chunk(3))
            ]
        else:
            weights = [("blocks.0.attn.qkv_proj.weight", source)]
        transformer.load_weights(weights, startup_lora=fusion)
        fusion.validate()
        assert merge_devices and all(device == torch.device("cpu") for device in merge_devices)
        torch.testing.assert_close(torch.cat(observed), expected, rtol=0, atol=0)
        rows = 2 // size
        local = torch.cat([expected[i * 2 + rank * rows : i * 2 + (rank + 1) * rows] for i in range(3)])
        torch.testing.assert_close(layer.weight, local, rtol=0, atol=0)
        assert block.attn.qkv_proj is layer


@pytest.mark.parametrize("dtype", [torch.int8, torch.float8_e4m3fn])
def test_startup_rejects_prequantized_weights(dtype):
    model, _, _ = _adapter()
    fusion = MiniMaxH3StartupLoRA(model, 1.0)
    with pytest.raises(ValueError, match="original FP16/BF16/FP32"):
        fusion.merge("blocks.0.attn.qkv_proj.weight", torch.ones(6, 4).to(dtype))


def test_startup_rejects_unconsumed_adapter_targets():
    model, _, _ = _adapter()
    with pytest.raises(ValueError, match="targets were not loaded"):
        MiniMaxH3StartupLoRA(model, 1.0).validate()


def test_startup_config_supports_online_quantization_and_preserves_runtime_default(tmp_path):
    config = OmniDiffusionConfig(
        model=str(tmp_path),
        num_gpus=1,
        lora_path="adapter",
        lora_application="startup-merge",
        quantization_config={"transformer": {"method": "mxfp8"}},
    )
    assert config.lora_application == "startup-merge"
    assert OmniDiffusionConfig(model=str(tmp_path), num_gpus=1).lora_application == "runtime"


@pytest.mark.parametrize(
    "options,message",
    [
        ({"lora_path": None}, "exactly one"),
        ({"lora_path": ["a", "b"]}, "exactly one"),
        ({"lora_scale": float("nan")}, "finite"),
        ({"lora_backend": "distill"}, "requires lora_backend"),
        ({"host_weight_runtime_mode": "preferred"}, "host weight artifacts"),
        ({"enable_cpu_offload": True}, "only supports distributed layerwise offload"),
        ({"enable_layerwise_offload": True}, "only supports distributed layerwise offload"),
        (
            {"enable_distributed_layerwise_offload": True, "step_execution": True},
            "requires request-mode generation",
        ),
    ],
)
def test_startup_config_rejects_invalid_modes(options, message, tmp_path):
    kwargs = {"model": str(tmp_path), "num_gpus": 1, "lora_path": "adapter", "lora_application": "startup-merge"}
    kwargs.update(options)
    with pytest.raises(ValueError, match=message):
        OmniDiffusionConfig(**kwargs)


@pytest.mark.parametrize(
    "offload_options,uses_allgather",
    [
        ({"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False}, False),
        (
            {
                "diffusion_offload_config": {
                    "mode": "layer",
                    "components": ["dit"],
                    "layer_options": {"dit": {"weight_transfer": "allgather"}},
                }
            },
            True,
        ),
    ],
)
def test_startup_config_accepts_dlo(offload_options, uses_allgather, tmp_path):
    from vllm_omni.diffusion.offloader.config import OffloadStrategy, resolve_offload

    config = OmniDiffusionConfig(
        model=str(tmp_path),
        num_gpus=2,
        lora_path="adapter",
        lora_application="startup-merge",
        quantization_config={"transformer": {"method": "mxfp8"}},
        **offload_options,
    )
    policy = resolve_offload(config)
    assert policy.strategy is OffloadStrategy.DISTRIBUTED_LAYER_WISE
    assert policy.uses_allgather("dit") is uses_allgather


@pytest.mark.parametrize("mode", ["module", "layer"])
def test_startup_config_allows_text_encoder_only_offload(mode, tmp_path):
    from vllm_omni.diffusion.offloader.config import resolve_offload

    config = OmniDiffusionConfig(
        model=str(tmp_path),
        num_gpus=1,
        lora_path="adapter",
        lora_application="startup-merge",
        diffusion_offload_config={"mode": mode, "components": ["text_encoder"]},
    )

    policy = resolve_offload(config)
    assert policy.offloads("text_encoder")
    assert not policy.offloads("dit")


def test_startup_rejects_combined_partition_before_loading_adapter():
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    pipeline.partition = "combined"

    with pytest.raises(ValueError, match="requires one MiniMax-H3 partition"):
        pipeline.prepare_startup_lora()


def test_fixed_native_adapter_keeps_sampling_schedule_without_request_lora(tmp_path):
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model=str(tmp_path), num_gpus=1)
    pipeline._startup_lora_request = LoRARequest("fixed", 37, "unused")
    pipeline._startup_lora_scale = 0.8
    pipeline._native_lora_adapter_ids = {37}
    schedule = DMD2SigmaSchedule.from_positions([1.0, 0.9, 0.6, 0.2, 0.0])
    pipeline._lora_sigma_schedules = {37: schedule}
    pipeline._turbo_lora_specs = {}
    sampling = OmniDiffusionSamplingParams(num_inference_steps=4)
    assert pipeline._has_active_native_lora(sampling)
    pipeline._validate_native_sampling(sampling, task="t2va")
    assert pipeline._resolve_sigma_positions("t2va", sampling) == (schedule.base_schedule, 4)
    with pytest.raises(OmniClientError, match="fixed LoRA"):
        pipeline._has_active_native_lora(OmniDiffusionSamplingParams(lora_request=LoRARequest("other", 39, "unused")))


def test_invalid_or_overflowing_factors_fail():
    model, _, _ = _adapter()
    with pytest.raises(ValueError, match="finite"):
        MiniMaxH3StartupLoRA(model, float("inf"))
    weights = next(iter(model.loras.values()))
    weights.lora_b.fill_(float("inf"))
    with pytest.raises(ValueError, match="Non-finite"):
        MiniMaxH3StartupLoRA(model, 1.0)


def test_startup_peft_directory_uses_original_alpha_and_no_runtime_manager(tmp_path):
    import json

    from safetensors.torch import save_file

    _, a, b = _adapter()
    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 2, "lora_alpha": 3, "target_modules": ["qkv_proj"]}))
    save_file(
        {
            "base_model.model.transformer.blocks.0.attn.qkv_proj.lora_A.weight": a,
            "base_model.model.transformer.blocks.0.attn.qkv_proj.lora_B.weight": b,
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        model=str(tmp_path), num_gpus=1, lora_path=str(tmp_path), lora_scale=0.7, lora_application="startup-merge"
    )
    pipeline.partition = "fl2va"
    pipeline._turbo_lora_specs = {}
    pipeline._native_lora_adapter_ids = set()
    pipeline._lora_sigma_schedules = {}
    pipeline.prepare_startup_lora()
    assert pipeline.lora_is_fused
    fusion = pipeline._startup_lora_fusion
    original = torch.zeros(6, 4)
    expected = torch.addmm(original, b, a, alpha=1.05)
    actual = fusion.merge("blocks.0.attn.qkv_proj.weight", original)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    fusion.validate()
    assert pipeline._request_lora_selection(OmniDiffusionSamplingParams())[1] == 0.7


def test_packed_mlp_keeps_gate_and_up_scaling_independent(monkeypatch):
    from vllm.lora.lora_weights import PackedLoRALayerWeights

    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)

    transformer = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(transformer)
    block = torch.nn.Module()
    block.mlp = torch.nn.Module()
    with set_current_vllm_config(VllmConfig()):
        layer = linear.MergedColumnParallelLinear(4, [3, 3], bias=False, params_dtype=torch.float32)
    block.mlp.fc1 = layer
    transformer.blocks = torch.nn.ModuleList([block])

    name = "transformer.blocks.0.mlp.fc1"
    a = [torch.full((2, 4), 0.25), torch.full((2, 4), -0.5)]
    b = [torch.full((3, 2), 0.75), torch.full((3, 2), 1.25)]
    scales = [0.25, 1.75]
    weights = PackedLoRALayerWeights(name, 2, [1, 1], a, b, scaling=scales)
    fusion = MiniMaxH3StartupLoRA(LoRAModel(1, 2, {name: weights}), 0.7)
    source = torch.arange(24).float().reshape(6, 4) / 16
    transformer.load_weights([("blocks.0.mlp.fc1.weight", source)], startup_lora=fusion)
    expected = torch.cat(
        [part + (up @ down) * strength * 0.7 for part, down, up, strength in zip(source.chunk(2), a, b, scales)]
    )
    torch.testing.assert_close(layer.weight, expected)
    fusion.validate()


def test_startup_does_not_reuse_teacher_adaln_sidecar(monkeypatch, tmp_path):
    import vllm.distributed

    from vllm_omni.diffusion.models.minimax_h3 import adaln_cache

    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(
        model=str(tmp_path),
        num_gpus=1,
        lora_path="adapter",
        lora_application="startup-merge",
        cache_config={"minimax_h3_adaln_cache_path": str(tmp_path / "teacher")},
    )
    transformer = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(transformer)
    transformer.arch = MiniMaxH3DiTArchConfig()
    transformer.adaln_cache = adaln_cache.MiniMaxH3RuntimeAdalnCache(max_bytes=1024)
    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)

    def reject_teacher(*args, **kwargs):
        raise AssertionError("the teacher sidecar cannot describe startup-fused weights")

    monkeypatch.setattr(adaln_cache, "MiniMaxH3AdalnCache", reject_teacher)
    pipeline._configure_adaln_sidecar(transformer, "minimax_h3_adaln_cache_path", "fl2va", None, eligible=True)
    assert transformer.adaln_cache.sidecar is None
    assert getattr(transformer, "_adaln_sidecar_candidate", None) is None


def test_startup_retains_fp32_delta_for_bf16_source_and_fp32_target():
    from vllm.model_executor.model_loader.weight_utils import default_weight_loader

    name = "transformer.blocks.0.adaln_proj.linear"
    a, b = torch.full((2, 4), 0.125), torch.full((6, 2), 0.001)
    adapter = LoRAModel(1, 2, {name: LoRALayerWeights(name, 2, 2, a, b)})
    fusion = MiniMaxH3StartupLoRA(adapter, 1.0)
    source = torch.ones(6, 4, dtype=torch.bfloat16)
    parameter = torch.nn.Parameter(torch.empty(6, 4, dtype=torch.float32), requires_grad=False)
    fusion.load_weight("blocks.0.adaln_proj.linear.weight", default_weight_loader, parameter, source)
    expected = torch.addmm(source.float(), b, a)
    torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
    assert torch.all(parameter > source.float())
    assert torch.all(parameter.bfloat16() == source)
    fusion.validate()


def _adaln_warmup_pipeline(monkeypatch, tmp_path, *, mode="t2va", device=torch.device("cpu"), vllm_config=None):
    import vllm.distributed

    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import MiniMaxH3RuntimeAdalnCache
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_blocks import MiniMaxH3AdalnProj
    from vllm_omni.diffusion.models.minimax_h3.minimax_h3_transformer import MiniMaxH3TimeEmbedder

    monkeypatch.setattr(vllm.distributed, "get_tensor_model_parallel_world_size", lambda: 1)
    for module in (linear, parameter):
        monkeypatch.setattr(module, "get_tensor_model_parallel_world_size", lambda: 1)
        monkeypatch.setattr(module, "get_tensor_model_parallel_rank", lambda: 0)
    pipeline = MiniMaxH3Pipeline.__new__(MiniMaxH3Pipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.od_config = OmniDiffusionConfig(model=str(tmp_path), num_gpus=1)
    pipeline._startup_lora_request = LoRARequest("fixed", 37, "unused")
    pipeline._startup_lora_scale = 0.7
    pipeline._startup_lora_fusion = None
    pipeline._native_lora_adapter_ids = {37} if mode == "t2va" else set()
    pipeline._lora_sigma_schedules = {37: DMD2SigmaSchedule.from_positions([1.0, 0.7, 0.4, 0.15, 0.0])}
    pipeline._base_schedule_by_partition = {}
    pipeline._turbo_lora_specs = {}
    pipeline.default_video_shift, pipeline.default_audio_shift = 12.0, 3.0
    pipeline.partition = "ref2va" if mode.startswith("ref2va") else "fl2va"
    transformer = MiniMaxH3DiTModel.__new__(MiniMaxH3DiTModel)
    torch.nn.Module.__init__(transformer)
    transformer.arch = MiniMaxH3DiTArchConfig(
        num_layers=1, hidden_size=4, timestep_input_dim=4, time_embed_hidden_size=8, time_embed_dim=4
    )
    transformer.adaln_cache = MiniMaxH3RuntimeAdalnCache()
    with set_current_vllm_config(vllm_config or VllmConfig()), torch.device(device), torch.inference_mode(False):
        transformer.time_embedder = MiniMaxH3TimeEmbedder(transformer.arch, prefix="time_embedder")
        transformer.blocks = torch.nn.ModuleList([torch.nn.Module()])
        transformer.final_layer = torch.nn.Module()
        for owner, ratio, modalities, prefix in (
            (transformer.blocks[0], 6, 3, "blocks.0"),
            (transformer.final_layer, 2, 1, "final_layer"),
        ):
            owner.adaln_proj = MiniMaxH3AdalnProj(
                transformer.arch,
                ratio * modalities * 4,
                None,
                expand_ratio=ratio,
                modality_num=modalities,
                prefix=prefix + ".adaln_proj",
                adaln_cache=transformer.adaln_cache,
            )
        with torch.no_grad():
            for index, value in enumerate(transformer.parameters()):
                value.copy_(torch.linspace(-0.3, 0.3, value.numel(), device=device).reshape(value.shape) + index / 100)
    pipeline.transformer = transformer
    return pipeline


def _warmup_plans(pipeline, mode="t2va", *, video_shift=12.0):
    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import schedule_contract, timestep_plans

    return timestep_plans(
        schedule_contract(
            mode=mode,
            num_steps=4,
            base_schedule=[1.0, 0.7, 0.4, 0.15, 0.0],
            flow_shift=video_shift,
            audio_flow_shift=3.0,
        )
    )


@pytest.mark.parametrize("inference_startup", [False, True])
def test_startup_warmup_reuses_fused_adaln_on_first_request(monkeypatch, inference_startup, tmp_path):
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path)
    transformer = pipeline.transformer
    projection = transformer.blocks[0].adaln_proj
    original = projection.linear.weight.detach().clone()
    adapter, a, b = _adapter("transformer.blocks.0.adaln_proj.linear", rows=72)
    fused = MiniMaxH3StartupLoRA(adapter, 0.7).merge("blocks.0.adaln_proj.linear.weight", original)
    with torch.no_grad():
        projection.linear.weight.copy_(fused)
    assert not torch.equal(original, fused)
    calls = []
    projections = [projection, transformer.final_layer.adaln_proj]
    for owner in projections:
        forward = owner.linear.forward

        def record(x, *, original_forward=forward):
            calls.append(1)
            return original_forward(x)

        monkeypatch.setattr(owner.linear, "forward", record)
    with torch.inference_mode(inference_startup):
        pipeline.warmup_startup_adaln_cache()
    warmed = len(calls)
    assert warmed == 8
    cache = transformer.adaln_cache
    pipeline._prepare_adaln_adapter(OmniDiffusionSamplingParams(num_inference_steps=4))
    with torch.inference_mode():
        for plan in _warmup_plans(pipeline):
            embedding = transformer.time_embedder(plan)
            cache.prepare(embedding)
            for owner in projections:
                actual = torch.cat(owner(embedding), dim=-1).reshape(len(plan), -1)
                expected = torch.nn.functional.linear(
                    torch.nn.functional.silu(embedding).bfloat16(), owner.linear.weight, owner.linear.bias
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(calls) == warmed
    assert cache.hits == 8 and cache.misses == 8


@pytest.mark.parametrize("mode", ["fl2va", "ref2va-image", "ref2va-audio", "ref2va-mixed"])
def test_startup_warmup_covers_conditioning_timestep_plans(monkeypatch, mode, tmp_path):
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path, mode=mode)
    pipeline.warmup_startup_adaln_cache()
    cache = pipeline.transformer.adaln_cache
    misses = cache.misses
    with torch.inference_mode():
        for plan in _warmup_plans(pipeline, mode):
            embedding = pipeline.transformer.time_embedder(plan)
            cache.prepare(embedding)
            pipeline.transformer.blocks[0].adaln_proj(embedding)
            pipeline.transformer.final_layer.adaln_proj(embedding)
    assert cache.misses == misses
    assert cache.hits >= 8


def test_startup_warmup_invalidates_changed_weights_and_inputs(monkeypatch, tmp_path):
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path)
    pipeline.warmup_startup_adaln_cache()
    transformer, plan = pipeline.transformer, _warmup_plans(pipeline)[1]
    cache = transformer.adaln_cache
    with torch.no_grad():
        transformer.blocks[0].adaln_proj.linear.weight.add_(0.5)
        embedding = transformer.time_embedder(plan)
        cache.prepare(embedding)
        misses = cache.misses
        transformer.blocks[0].adaln_proj(embedding)
        transformer.final_layer.adaln_proj(embedding)
        assert cache.misses == misses + 1
        changed = transformer.time_embedder(_warmup_plans(pipeline, video_shift=6.0)[1])
        cache.prepare(changed)
        transformer.blocks[0].adaln_proj(changed)
        transformer.final_layer.adaln_proj(changed)
        assert cache.misses == misses + 3


def test_startup_warmup_respects_cache_disable_and_fusion_lifecycle(monkeypatch, tmp_path):
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path)
    pipeline.transformer.adaln_cache.max_bytes = 0
    pipeline.warmup_startup_adaln_cache()
    assert pipeline.transformer.adaln_cache.misses == 0
    pipeline._startup_lora_fusion = object()
    with pytest.raises(RuntimeError, match="completed startup LoRA fusion"):
        pipeline.warmup_startup_adaln_cache()
    pipeline._startup_lora_request = None
    pipeline.warmup_startup_adaln_cache()


@pytest.mark.parametrize(
    "application,load_format,expected",
    [
        ("startup-merge", "default", ["load", "warm", "synchronize"]),
        ("runtime", "default", ["load", "synchronize"]),
        ("startup-merge", "dummy", ["load", "synchronize"]),
    ],
)
def test_worker_warms_fixed_adaln_after_final_runner_setup(monkeypatch, application, load_format, expected, tmp_path):
    from contextlib import nullcontext

    import vllm_omni.diffusion.worker.diffusion_worker as worker_module
    import vllm_omni.distributed.weight_sharing.worker_session as sharing
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path)
    worker = worker_module.DiffusionWorker.__new__(worker_module.DiffusionWorker)
    worker.od_config = OmniDiffusionConfig(
        model=str(tmp_path), num_gpus=1, lora_path="unused", lora_application=application
    )
    worker.vllm_config = VllmConfig()
    worker.rank = worker.local_rank = 0
    worker.lora_manager = None
    runner = DiffusionModelRunner.__new__(DiffusionModelRunner)
    runner.pipeline = pipeline
    worker.model_runner = runner
    events = []
    monkeypatch.setattr(runner, "load_model", lambda **kwargs: events.append("load"))
    monkeypatch.setattr(pipeline, "warmup_startup_adaln_cache", lambda: events.append("warm"))
    monkeypatch.setattr(worker_module, "set_forward_context", lambda **kwargs: nullcontext())
    monkeypatch.setattr(worker_module.current_omni_platform, "synchronize", lambda: events.append("synchronize"))
    monkeypatch.setattr(worker_module.current_omni_platform, "current_device", lambda: 0)
    monkeypatch.setattr(worker_module, "get_process_gpu_memory", lambda rank: None)
    monkeypatch.setattr(sharing, "install_runner_session", lambda *args, **kwargs: None)
    worker.load_model(load_format=load_format)
    assert events == expected


@pytest.mark.parametrize(
    "offload_options",
    [
        {"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False},
        {"diffusion_offload_config": {"mode": "module", "components": ["text_encoder"]}},
        {"diffusion_offload_config": {"mode": "layer", "components": ["text_encoder"]}},
    ],
)
def test_startup_adaln_cache_defers_prewarm_with_weight_offload(monkeypatch, offload_options, tmp_path):
    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path)
    pipeline.od_config = OmniDiffusionConfig(model=str(tmp_path), num_gpus=1, **offload_options)

    def unexpected_projection(*args, **kwargs):
        pytest.fail("startup must not call the time embedder ahead of weight placement")

    monkeypatch.setattr(pipeline.transformer.time_embedder, "forward", unexpected_projection)
    pipeline.warmup_startup_adaln_cache()
    assert pipeline.transformer.adaln_cache.misses == 0


def test_startup_warmup_uses_turbo_required_steps_and_shifts(monkeypatch, tmp_path):
    from vllm_omni.diffusion.models.minimax_h3.adaln_cache import schedule_contract, timestep_plans
    from vllm_omni.diffusion.models.minimax_h3.lora import TurboSpec

    pipeline = _adaln_warmup_pipeline(monkeypatch, tmp_path, mode="fl2va")
    pipeline._lora_sigma_schedules = {}
    pipeline._turbo_lora_specs = {
        37: TurboSpec("test", "fl2v", denoise_steps=3, video_shift=6.0, audio_shift=2.0, rank=2, alpha=2)
    }
    pipeline.warmup_startup_adaln_cache()
    cache = pipeline.transformer.adaln_cache
    misses = cache.misses
    plans = timestep_plans(
        schedule_contract(mode="fl2va", num_steps=3, base_schedule=None, flow_shift=6.0, audio_flow_shift=2.0)
    )
    with torch.inference_mode():
        for plan in plans:
            embedding = pipeline.transformer.time_embedder(plan)
            cache.prepare(embedding)
            pipeline.transformer.blocks[0].adaln_proj(embedding)
    assert cache.misses == misses


@pytest.mark.parametrize("cube_value", [None, 3])
@pytest.mark.parametrize("allow_hf32", [False, True])
def test_npu_fusion_precision_guard_restores_original_policy(monkeypatch, cube_value, allow_hf32):
    import sys
    from dataclasses import dataclass
    from enum import IntEnum
    from types import ModuleType

    from vllm_omni.diffusion.models.minimax_h3.startup_lora import _npu_fp32_matmul

    class CubeMode(IntEnum):
        KEEP_DTYPE = 0
        USE_HF32 = 3

    @dataclass
    class MatmulPolicy:
        allow_hf32: bool
        cube_math_type: CubeMode | None

    previous_cube = None if cube_value is None else CubeMode(cube_value)
    policy = MatmulPolicy(allow_hf32, previous_cube)
    runtime = ModuleType("torch_npu")
    runtime.npu = ModuleType("npu")
    runtime.npu.matmul = policy
    monkeypatch.setitem(sys.modules, "torch_npu", runtime)
    with _npu_fp32_matmul():
        assert policy.allow_hf32 is False
        assert policy.cube_math_type == (None if previous_cube is None else CubeMode.KEEP_DTYPE)
    assert policy.allow_hf32 == allow_hf32 and policy.cube_math_type == previous_cube
    with pytest.raises(ValueError, match="fusion failure"):
        with _npu_fp32_matmul():
            raise ValueError("fusion failure")
    assert policy.allow_hf32 == allow_hf32 and policy.cube_math_type == previous_cube
