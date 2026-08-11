# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from vllm.config.load import LoadConfig

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture(autouse=True)
def _single_rank_cfg_state(monkeypatch):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video

    monkeypatch.setattr(pipeline_lingbot_video, "get_classifier_free_guidance_world_size", lambda: 1)


class _TinyComponent(nn.Module):
    def __init__(self, *, dtype: torch.dtype = torch.float32):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=dtype))


def _tiny_transformer_config() -> dict:
    return {
        "_class_name": "LingBotVideoTransformer3DModel",
        "patch_size": [1, 1, 1],
        "in_channels": 2,
        "out_channels": 2,
        "hidden_size": 16,
        "num_attention_heads": 1,
        "depth": 0,
        "intermediate_size": 32,
        "text_dim": 8,
        "freq_dim": 8,
        "axes_dims": [4, 4, 8],
        "axes_lens": [32, 32, 32],
    }


def _build_pipeline(
    mocker,
    *,
    refiner_enabled: bool = False,
    refiner_model: str | None = None,
    refiner_revision: str | None = None,
):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    model = "test-org/lingbot-video"
    revision = "native-loader-test-revision"
    subfolders = {
        "transformer_subfolder": "custom_transformer",
        "text_encoder_subfolder": "custom_text_encoder",
        "processor_subfolder": "custom_processor",
        "vae_subfolder": "custom_vae",
        "scheduler_subfolder": "custom_scheduler",
    }
    if refiner_enabled:
        subfolders["lingbot_refiner"] = {
            "enabled": True,
            "transformer_subfolder": "custom_refiner",
        }
        if refiner_model is not None:
            subfolders["lingbot_refiner"]["model_dir"] = refiner_model
        if refiner_revision is not None:
            subfolders["lingbot_refiner"]["revision"] = refiner_revision
    od_config = SimpleNamespace(
        model=model,
        revision=revision,
        dtype=torch.bfloat16,
        model_config=subfolders,
        quantization_config=None,
        parallel_config=SimpleNamespace(use_hsdp=False),
    )

    mocker.patch.object(module, "get_local_device", return_value=torch.device("cpu"))
    prefetch = mocker.patch.object(module, "prefetch_subfolders")
    load_config = mocker.patch.object(
        module.LingBotVideoTransformer3DModel,
        "load_config",
        return_value=_tiny_transformer_config(),
    )
    text_encoder_load = mocker.patch.object(
        module.Qwen3VLForConditionalGeneration,
        "from_pretrained",
        return_value=_TinyComponent(dtype=torch.bfloat16),
    )
    processor_load = mocker.patch.object(
        module.Qwen3VLProcessor,
        "from_pretrained",
        return_value=SimpleNamespace(),
    )
    vae_load = mocker.patch.object(
        module.AutoencoderKLWan,
        "from_pretrained",
        return_value=_TinyComponent(dtype=torch.float32),
    )
    scheduler_load = mocker.patch.object(
        module.FlowUniPCMultistepScheduler,
        "from_pretrained",
        side_effect=lambda *args, **kwargs: SimpleNamespace(
            model=args[0],
            subfolder=kwargs["subfolder"],
        ),
    )

    with torch.device("cpu"):
        pipeline = module.LingBotVideoPipeline(od_config=od_config)

    return pipeline, SimpleNamespace(
        model=model,
        revision=revision,
        subfolders=subfolders,
        prefetch=prefetch,
        load_config=load_config,
        text_encoder_load=text_encoder_load,
        processor_load=processor_load,
        vae_load=vae_load,
        scheduler_load=scheduler_load,
        od_config=od_config,
    )


def test_constructor_uses_native_transformer_source_and_component_revision(mocker):
    pipeline, calls = _build_pipeline(mocker)

    component_subfolders = list(calls.subfolders.values())
    assert not hasattr(type(pipeline.transformer), "from_pretrained")
    calls.prefetch.assert_called_once_with(
        calls.model,
        local_files_only=False,
        subfolders=component_subfolders,
        revision=calls.revision,
    )
    calls.load_config.assert_called_once_with(
        calls.model,
        subfolder=calls.subfolders["transformer_subfolder"],
        revision=calls.revision,
        local_files_only=False,
    )
    calls.text_encoder_load.assert_called_once_with(
        calls.model,
        subfolder=calls.subfolders["text_encoder_subfolder"],
        local_files_only=False,
        dtype=torch.bfloat16,
        revision=calls.revision,
    )
    calls.processor_load.assert_called_once_with(
        calls.model,
        subfolder=calls.subfolders["processor_subfolder"],
        local_files_only=False,
        revision=calls.revision,
    )
    calls.vae_load.assert_called_once_with(
        calls.model,
        subfolder=calls.subfolders["vae_subfolder"],
        local_files_only=False,
        torch_dtype=torch.float32,
        revision=calls.revision,
    )
    calls.scheduler_load.assert_called_once_with(
        calls.model,
        subfolder=calls.subfolders["scheduler_subfolder"],
        local_files_only=False,
        revision=calls.revision,
    )

    [source] = pipeline.weights_sources
    assert source.model_or_path == calls.model
    assert source.subfolder == calls.subfolders["transformer_subfolder"]
    assert source.revision == calls.revision
    assert source.prefix == "transformer."
    assert source.fall_back_to_pt is True
    assert pipeline.transformer._layerwise_offload_blocks_attrs == ["blocks"]
    assert pipeline.device == torch.device("cpu")
    assert next(pipeline.transformer.parameters()).device.type == "cpu"
    assert next(pipeline.text_encoder.parameters()).device.type == "cpu"
    assert next(pipeline.vae.parameters()).device.type == "cpu"


def test_native_weight_stream_is_strict_and_preserves_mixed_dtype(mocker):
    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    pipeline, calls = _build_pipeline(mocker)
    checkpoint: dict[str, torch.Tensor] = {}
    for index, (name, parameter) in enumerate(pipeline.transformer.named_parameters()):
        values = torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        checkpoint[f"transformer.{name}"] = values / 1000 + 0.1234567 + index

    loaded = pipeline.load_weights(checkpoint.items())

    assert loaded == set(checkpoint)
    loader = DiffusersPipelineLoader(LoadConfig(), calls.od_config)
    assert loader._get_expected_parameter_names(pipeline) == set(checkpoint)
    assert all(not name.startswith(("text_encoder.", "vae.")) for name in loaded)

    params = dict(pipeline.named_parameters())
    bulk_name = "transformer.patch_embedder.weight"
    fp32_name = "transformer.time_embedder.linear_1.weight"
    assert params[bulk_name].dtype == torch.bfloat16
    assert params[fp32_name].dtype == torch.float32
    assert torch.equal(params[bulk_name], checkpoint[bulk_name].to(torch.bfloat16))
    assert torch.equal(params[fp32_name], checkpoint[fp32_name])


def test_pipeline_dtype_move_preserves_component_specific_policies(mocker):
    pipeline, _ = _build_pipeline(mocker)
    pipeline.text_encoder.to(dtype=torch.float16)

    pipeline.to(dtype=torch.bfloat16)

    assert pipeline.text_encoder.weight.dtype == torch.float16
    assert pipeline.vae.weight.dtype == torch.float32
    assert pipeline.transformer.patch_embedder.weight.dtype == torch.bfloat16
    assert pipeline.transformer.time_embedder.linear_1.weight.dtype == torch.float32


class _DeviceTrackingVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.float32))
        self.current_device = torch.device("cuda:0")
        self.moves: list[torch.device] = []

    def to(self, *args, **kwargs):
        device, _, _, _ = torch._C._nn._parse_to(*args, **kwargs)
        if device is not None:
            self.current_device = torch.device(device)
            self.moves.append(self.current_device)
        return self


class _FailingTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros((), dtype=torch.bfloat16))
        self.config = SimpleNamespace(in_channels=1)

    def forward(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("denoise failed")


class _PassingTransformer(_FailingTransformer):
    def forward(self, hidden_states, *args, **kwargs):
        del args, kwargs
        return (torch.zeros_like(hidden_states),)


class _OneStepScheduler:
    sigma_max = 1.0
    sigma_min = 0.0

    def set_timesteps(self, num_inference_steps, *, device, shift, **kwargs):
        del num_inference_steps, shift, kwargs
        self.timesteps = torch.tensor([1000.0], device=device)

    def step(self, noise_pred, timestep, latents, **kwargs):
        del noise_pred, timestep, kwargs
        return (latents,)


def _build_generation_pipeline(transformer: nn.Module):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    pipeline = object.__new__(module.LingBotVideoPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae = _DeviceTrackingVAE()
    pipeline.transformer = transformer
    pipeline.scheduler = _OneStepScheduler()
    pipeline.progress_bar = lambda timesteps: timesteps
    pipeline.encode_prompt = lambda *args, **kwargs: (
        torch.ones(1, 2, 4),
        torch.ones(1, 2, dtype=torch.long),
    )
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 1, 1, 1)
    return pipeline


def _generation_kwargs(*, output_type: str = "pt") -> dict:
    from vllm_omni.diffusion.models.lingbot_video import (
        LingBotExecutionOptions,
        LingBotGenerationMode,
    )

    return {
        "prompt": "a robot",
        "mode": LingBotGenerationMode.T2V,
        "height": 16,
        "width": 16,
        "num_frames": 1,
        "num_inference_steps": 1,
        "guidance_scale": 1.0,
        "shift": 3.0,
        "output_type": output_type,
        "execution_options": LingBotExecutionOptions(offload_vae_during_denoise=True),
    }


def _patch_tracked_vae_device(mocker, module):
    original_module_device = module._module_device
    mocker.patch.object(
        module,
        "_module_device",
        side_effect=lambda value: value.current_device
        if isinstance(value, _DeviceTrackingVAE)
        else original_module_device(value),
    )
    mocker.patch.object(torch.accelerator, "empty_cache")


def test_vae_is_restored_before_normal_decode(mocker):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    pipeline = _build_generation_pipeline(_PassingTransformer())
    decode_devices: list[torch.device] = []
    pipeline._decode_latents_internal = lambda latents: (
        decode_devices.append(pipeline.vae.current_device) or torch.zeros(1, 1, 1, 1, 1)
    )
    _patch_tracked_vae_device(mocker, module)

    pipeline._generate(**_generation_kwargs())

    assert pipeline.vae.moves == [torch.device("cpu"), torch.device("cuda:0")]
    assert decode_devices == [torch.device("cuda:0")]
    assert pipeline.vae.current_device == torch.device("cuda:0")


def test_latent_early_return_offloads_and_restores_vae(mocker):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    pipeline = _build_generation_pipeline(_PassingTransformer())
    _patch_tracked_vae_device(mocker, module)

    result = pipeline._generate(**_generation_kwargs(output_type="latent"))

    assert isinstance(result, torch.Tensor)
    assert pipeline.vae.moves == [torch.device("cpu"), torch.device("cuda:0")]
    assert pipeline.vae.current_device == torch.device("cuda:0")


def test_vae_is_restored_when_denoise_raises(mocker):
    from vllm_omni.diffusion.models.lingbot_video import pipeline_lingbot_video as module

    pipeline = _build_generation_pipeline(_FailingTransformer())
    _patch_tracked_vae_device(mocker, module)

    with pytest.raises(RuntimeError, match="denoise failed"):
        pipeline._generate(**_generation_kwargs())

    assert pipeline.vae.moves == [torch.device("cpu"), torch.device("cuda:0")]
    assert pipeline.vae.current_device == torch.device("cuda:0")


def test_constructor_builds_independent_native_refiner_source_and_scheduler(mocker):
    from vllm_omni.diffusion.offloader.module_collector import ModuleDiscovery

    pipeline, calls = _build_pipeline(mocker, refiner_enabled=True)

    assert pipeline._dit_modules == ["transformer", "refiner_transformer"]
    assert pipeline.refiner_transformer is not None
    assert pipeline.refiner_scheduler is not None
    assert pipeline.scheduler is not pipeline.refiner_scheduler
    assert calls.prefetch.call_args_list == [
        mocker.call(
            calls.model,
            local_files_only=False,
            subfolders=[
                "custom_transformer",
                "custom_text_encoder",
                "custom_processor",
                "custom_vae",
                "custom_scheduler",
                "custom_refiner",
            ],
            revision=calls.revision,
        )
    ]
    assert calls.load_config.call_args_list == [
        mocker.call(
            calls.model,
            subfolder="custom_transformer",
            revision=calls.revision,
            local_files_only=False,
        ),
        mocker.call(
            calls.model,
            subfolder="custom_refiner",
            revision=calls.revision,
            local_files_only=False,
        ),
    ]
    assert calls.scheduler_load.call_count == 2

    assert [(source.subfolder, source.prefix) for source in pipeline.weights_sources] == [
        ("custom_transformer", "transformer."),
        ("custom_refiner", "refiner_transformer."),
    ]
    discovered = ModuleDiscovery.discover(pipeline)
    assert discovered.dit_names == ["transformer", "refiner_transformer"]
    assert discovered.outermost_dits()[0] == ["transformer", "refiner_transformer"]


def test_constructor_prefetches_independent_refiner_revision(mocker):
    refiner_model = "test-org/lingbot-video-refiner"
    refiner_revision = "refiner-test-revision"

    _, calls = _build_pipeline(
        mocker,
        refiner_enabled=True,
        refiner_model=refiner_model,
        refiner_revision=refiner_revision,
    )

    assert calls.prefetch.call_args_list == [
        mocker.call(
            calls.model,
            local_files_only=False,
            subfolders=[
                "custom_transformer",
                "custom_text_encoder",
                "custom_processor",
                "custom_vae",
                "custom_scheduler",
            ],
            revision=calls.revision,
        ),
        mocker.call(
            refiner_model,
            local_files_only=False,
            subfolders=["custom_refiner", "custom_scheduler"],
            revision=refiner_revision,
        ),
    ]


def test_native_weight_stream_strictly_loads_base_and_refiner_prefixes(mocker):
    from vllm.config.load import LoadConfig

    from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader

    pipeline, calls = _build_pipeline(mocker, refiner_enabled=True)
    checkpoint = {
        name: torch.arange(parameter.numel(), dtype=torch.float32).reshape(parameter.shape)
        for name, parameter in pipeline.named_parameters()
        if name.startswith(("transformer.", "refiner_transformer."))
    }

    loaded = pipeline.load_weights(checkpoint.items())

    assert loaded == set(checkpoint)
    loader = DiffusersPipelineLoader(LoadConfig(), calls.od_config)
    assert loader._get_expected_parameter_names(pipeline) == set(checkpoint)
