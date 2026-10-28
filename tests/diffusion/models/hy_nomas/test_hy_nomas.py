# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys
from datetime import timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from vllm.config import LoadConfig

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.media import VideoValueRange
from vllm_omni.diffusion.model_loader.diffusers_loader import DiffusersPipelineLoader
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import MODEL_ID, REQUIRED_FILES
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import build_transformer, load_transformer_weights
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import (
    HyNomasPipeline,
    build_decoder,
    build_reference_config,
    validate_config,
)
from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import action_conditioning, normalize_request
from vllm_omni.diffusion.offloader import sequential_backend
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import (
    is_diffusion_model,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import get_extra_body_params, get_video_generation_defaults

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


TRANSFORMER_CONFIG = {
    "_class_name": "HyNomasTransformer3DModel",
    "in_channels": 32,
    "out_channels": 32,
    "num_layers": 54,
    "num_attention_heads": 16,
    "attention_head_dim": 128,
    "mlp_ratio": 4.0,
    "text_embed_dim": 3584,
    "rope_axes_dim": [16, 56, 56],
    "rope_theta": 256.0,
    "camera_embed_dim": 128,
    "camera_attention_dim": 1024,
    "camera_num_attention_heads": 16,
    "camera_rope_axes_dim": [8, 28, 28],
}
DECODER_CONFIG = {
    "_class_name": "PixelUMMDecoder",
    "hidden_size": 4096,
    "intermediate_size": 12288,
    "num_layers": 36,
    "num_heads": 32,
    "num_kv_heads": 8,
    "head_dim": 128,
    "rms_norm_eps": 1e-6,
    "rope_theta_t": 1000000.0,
    "rope_theta_hw": 10000.0,
    "local_attention_tile_pixels": 128,
    "denoising_step_list": [0.999, 0.866, 0.634, 0.342],
    "num_steps": 1,
    "seed": 42,
    "temporal_window_latents": 10,
    "temporal_overlap_latents": 5,
    "decoder_tile_pixels": 0,
    "decoder_tile_overlap_pixels": 64,
    "tail_hold_latents": 1,
}


@pytest.fixture
def checkpoint(tmp_path):
    configs = {
        "modular_model_index.json": {"_class_name": "HyNomasModularPipeline", "_diffusers_version": "0.42.0.dev0"},
        "transformer/config.json": TRANSFORMER_CONFIG,
        "decoder/config.json": DECODER_CONFIG,
        "scheduler/scheduler_config.json": {
            "_class_name": "FlowMatchEulerDiscreteScheduler",
            "shift": 5.0,
            "num_train_timesteps": 1000,
        },
    }
    for relative, config in configs.items():
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(config))
    return tmp_path


def test_old_reference_decoder_fails_before_component_loading(checkpoint, monkeypatch, mocker):
    class LegacySession:
        def __init__(self, model, *, to_cpu=True, retain_history=True):
            pass

    reference = ModuleType("hymm.models.vae.pixelumm_decoder.streaming")
    monkeypatch.setattr(reference, "WindowStreamingDecodeSession", LegacySession, raising=False)
    monkeypatch.setitem(sys.modules, reference.__name__, reference)
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", lambda: torch.device("cuda"))
    download = mocker.patch.object(pipeline_hy_nomas, "download_weights_from_hf_specific")
    config = OmniDiffusionConfig(
        model=str(checkpoint), dtype=torch.bfloat16, distributed_executor_backend="mp", enforce_eager=True
    )
    with pytest.raises(RuntimeError, match="pinned rolling PixelUMM revision"):
        pipeline_hy_nomas.HyNomasPipeline(od_config=config)
    download.assert_not_called()


def test_discovery_from_modular_index(checkpoint, monkeypatch):
    assert resolve_model_class_name(str(checkpoint)) == "HyNomasModularPipeline"
    assert is_diffusion_model(str(checkpoint))
    hub_api = Mock(side_effect=AssertionError("Local checkpoint configuration must not create a Hub client"))
    monkeypatch.setattr("vllm.transformers_utils.repo_utils.hf_api", hub_api)
    config = OmniDiffusionConfig(model=str(checkpoint))
    config.enrich_config()
    assert config.model_class_name == "HyNomasModularPipeline"
    assert get_diffusion_model_metadata(config.model_class_name).final_output_type == "video"
    hub_api.assert_not_called()


def test_discovery_does_not_import_runtime_registry(checkpoint):
    code = (
        "import sys; "
        "from vllm_omni.diffusion.utils.hf_utils import get_diffusion_model_index; "
        f"assert get_diffusion_model_index({str(checkpoint)!r})['_class_name'] == 'HyNomasModularPipeline'; "
        "assert 'vllm_omni.diffusion.registry' not in sys.modules; "
        "assert 'hymm' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    )
    assert result.returncode == 0, result.stderr


def test_hub_discovery_reads_standard_modular_index(monkeypatch):
    def read_config(filename, model, revision):
        assert model == MODEL_ID
        return {"_class_name": "HyNomasModularPipeline"} if filename == "modular_model_index.json" else None

    monkeypatch.setattr("vllm_omni.diffusion.utils.hf_utils.get_hf_file_to_dict", read_config)
    assert resolve_model_class_name(MODEL_ID) == "HyNomasModularPipeline"


def test_local_pipeline_index_keeps_precedence(checkpoint):
    (checkpoint / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline", "_diffusers_version": "0.40.0"})
    )
    assert resolve_model_class_name(str(checkpoint)) == "QwenImagePipeline"
    config = OmniDiffusionConfig(model=str(checkpoint))
    config.enrich_config()
    assert config.model_class_name == "QwenImagePipeline"


@pytest.mark.parametrize("architecture", ["HyNomasPipeline", "HyNomasModularPipeline"])
def test_public_defaults(architecture):
    defaults = get_video_generation_defaults(architecture)
    assert (defaults.height, defaults.width, defaults.num_frames) == (704, 1280, 121)
    assert (defaults.num_inference_steps, defaults.guidance_scale, defaults.fps) == (8, 1.0, 24.0)
    assert "segment_frame_boundaries" in get_extra_body_params(architecture)


@pytest.mark.parametrize("text_encoder_override", [None, "local-text-encoder"])
def test_pipeline_loads_diffusers_components(checkpoint, monkeypatch, text_encoder_override):
    class RollingSession:
        def __init__(self, model, *, to_cpu=True, retain_history=True, rolling=False):
            pass

    streaming = ModuleType("hymm.models.vae.pixelumm_decoder.streaming")
    monkeypatch.setattr(streaming, "WindowStreamingDecodeSession", RollingSession, raising=False)
    monkeypatch.setitem(sys.modules, streaming.__name__, streaming)
    download = Mock(return_value=str(checkpoint))
    transformer = nn.Linear(2, 2, bias=False)
    transformer.patch_size = (1, 1, 1)
    transformer.config = SimpleNamespace(in_channels=2)
    transformer.double_blocks = []
    transformer.heads_num = 1
    transformer.hidden_size = 2
    text_encoder = nn.Linear(2, 2)
    vae = nn.Linear(2, 2)
    vae.spatial_compression_ratio = 32
    loader = ModuleType("hymm.models.text_encoder")
    load_text = Mock(return_value=text_encoder)
    monkeypatch.setattr(loader, "TextEncoder", load_text, raising=False)
    constants = ModuleType("configs.constants")
    monkeypatch.setattr(constants, "PROMPT_TEMPLATE", {}, raising=False)
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, constants.__name__, constants)
    monkeypatch.setattr(pipeline_hy_nomas, "build_transformer", Mock(return_value=transformer))
    monkeypatch.setattr(pipeline_hy_nomas, "build_decoder", Mock(return_value=vae))
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(pipeline_hy_nomas, "initialize_reference_parallelism", Mock(return_value=None))
    monkeypatch.setattr(pipeline_hy_nomas, "download_weights_from_hf_specific", download)
    monkeypatch.setattr(
        pipeline_hy_nomas,
        "build_reference_config",
        Mock(
            return_value=DictConfig(
                {
                    "inference": {"cache_noise_sigma": 0.206897},
                    "text_encoder_type": "llm",
                    "text_encoder": {
                        "tokenizer": "llm",
                        "text_len": 1280,
                        "use_ll_cap_dataset": False,
                        "text_encoder_precision": "bf16",
                        "prompt_template": None,
                        "prompt_template_video": None,
                    },
                }
            )
        ),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    config = OmniDiffusionConfig(
        model=MODEL_ID,
        revision="test-revision",
        dtype=torch.bfloat16,
        distributed_executor_backend="mp",
        custom_pipeline_args={} if text_encoder_override is None else {"text_encoder": text_encoder_override},
    )
    pipeline = HyNomasPipeline(od_config=config)
    assert pipeline.transformer is transformer and pipeline.text_encoder is text_encoder and pipeline.vae is vae
    assert [(source.subfolder, source.prefix) for source in pipeline.weights_sources] == [
        ("transformer", "transformer."),
        ("decoder", "vae."),
    ]
    assert all(
        source.model_or_path == MODEL_ID and source.revision == "test-revision" for source in pipeline.weights_sources
    )
    assert all(source.allow_patterns_overrides is None for source in pipeline.weights_sources)
    assert load_text.call_args.kwargs["text_encoder_path"] == (
        text_encoder_override or str(checkpoint / "text_encoder")
    )
    assert load_text.call_args.kwargs["tokenizer_path"] == (text_encoder_override or str(checkpoint / "tokenizer"))
    download.assert_called_once_with(
        MODEL_ID,
        cache_dir=None,
        allow_patterns=[*REQUIRED_FILES, *(["text_encoder/*", "tokenizer/*"] if text_encoder_override is None else [])],
        revision="test-revision",
        require_all=True,
    )


@pytest.mark.parametrize("encoder", ["", " ", 42, None])
def test_invalid_encoder_fails_before_download(monkeypatch, encoder):
    download = Mock()
    monkeypatch.setattr(pipeline_hy_nomas, "download_weights_from_hf_specific", download)
    config = OmniDiffusionConfig(
        dtype=torch.bfloat16, distributed_executor_backend="mp", custom_pipeline_args={"text_encoder": encoder}
    )
    with pytest.raises(ValueError, match="text_encoder must be"):
        HyNomasPipeline(od_config=config)
    download.assert_not_called()


@pytest.mark.parametrize("duration,frames", [(4, 97), (8, 193), (0.5, 13)])
def test_duration_resolves_frame_count(duration, frames):
    params = OmniDiffusionSamplingParams(extra_args={"duration": duration, "generate_sound": False})
    assert normalize_request("A forest", params).num_frames == frames
    params.num_frames = frames + 4
    with pytest.raises(OmniClientError, match="conflicts with duration"):
        normalize_request("A forest", params)


@pytest.mark.parametrize("duration", [0, -1, True, "bad", float("inf"), 1.1])
def test_invalid_duration(duration):
    with pytest.raises(OmniClientError, match="duration"):
        normalize_request("A forest", OmniDiffusionSamplingParams(extra_args={"duration": duration}))


@pytest.mark.parametrize("as_list", [False, True])
def test_generator_seed_from_shared_video_example(as_list):
    generator = torch.Generator().manual_seed(123)
    params = OmniDiffusionSamplingParams(num_frames=9, generator=[generator] if as_list else generator)
    assert normalize_request("A forest", params).seed == 123


def test_segment_and_camera_request():
    params = OmniDiffusionSamplingParams(
        num_frames=121,
        seed=17,
        extra_args={
            "segment_frame_boundaries": [0, 60, 121],
            "camera_trajectory": "forward",
            "segment_has_camera": [False, True],
        },
    )
    request = normalize_request("A forest. ||| A river appears.", params)
    assert request.prompts == ["A forest.", "A river appears."]
    assert request.boundaries == [0, 60, 121]
    assert request.segment_has_camera == [False, True]
    assert request.seed == 17
    assert request.steps == 8
    assert request.camera_trajectory == "forward"


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"num_frames": 0}, "num_frames"),
        ({"num_frames": 120}, "num_frames"),
        ({"height": 720}, "height"),
        ({"width": 0}, "width"),
        ({"num_inference_steps": 0}, "steps"),
        ({"guidance_scale": 6.0}, "guidance_scale"),
        ({"extra_args": {"flow_shift": 3.0}}, "flow_shift"),
        ({"fps": 30}, "24 fps"),
        ({"num_outputs_per_prompt": 2}, "one output"),
        ({"output_type": "latent"}, "decoded video"),
        ({"extra_args": {"unknown": 1}}, "Unsupported"),
        ({"extra_args": {"segment_frame_boundaries": [0, 120]}}, "boundaries"),
        ({"extra_args": {"segment_has_camera": [True]}}, "segment_has_camera"),
        ({"extra_args": {"camera_trajectory": ""}}, "camera_trajectory"),
        ({"extra_args": {"generate_sound": True}}, "without audio"),
    ],
)
def test_invalid_sampling(overrides, match):
    params = OmniDiffusionSamplingParams(**({"num_frames": 121} | overrides))
    with pytest.raises(OmniClientError, match=match):
        normalize_request("A forest", params)


@pytest.mark.parametrize("boundaries", [None, [0, 61, 121], [0, 0, 121], [0, 121], [0, True, 121]])
def test_invalid_segments(boundaries):
    params = OmniDiffusionSamplingParams(num_frames=121)
    if boundaries is not None:
        params.extra_args["segment_frame_boundaries"] = boundaries
    with pytest.raises(OmniClientError):
        normalize_request("A forest ||| A river", params)


@pytest.mark.parametrize("prompt", ["", "one ||| ", {"prompt": "x", "multi_modal_data": {"image": "x"}}])
def test_invalid_prompt(prompt):
    with pytest.raises(OmniClientError):
        normalize_request(prompt, OmniDiffusionSamplingParams(num_frames=121))


@pytest.mark.parametrize("negative_prompt", [None, ""])
def test_empty_negative_prompt_is_accepted(negative_prompt):
    request = normalize_request(
        {"prompt": "A forest", "negative_prompt": negative_prompt}, OmniDiffusionSamplingParams()
    )
    assert request.prompts == ["A forest"]


@pytest.mark.parametrize("negative_prompt", ["blur", " "])
def test_nonempty_negative_prompt_is_rejected(negative_prompt):
    with pytest.raises(OmniClientError, match="does not support negative_prompt"):
        normalize_request({"prompt": "A forest", "negative_prompt": negative_prompt}, OmniDiffusionSamplingParams())


@pytest.mark.parametrize("negative_prompt", [False, 1, [], {}])
def test_negative_prompt_requires_string(negative_prompt):
    with pytest.raises(OmniClientError, match="negative_prompt must be a string"):
        normalize_request({"prompt": "A forest", "negative_prompt": negative_prompt}, OmniDiffusionSamplingParams())


@pytest.fixture
def pipeline(monkeypatch):
    from tests.diffusion.models.hy_nomas.test_ar import PipelineWithCPUComponents

    module = ModuleType("hymm.models.dit.modules.attention.range_attention")
    monkeypatch.setattr(module, "clear_range_cache", Mock(), raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pipeline = PipelineWithCPUComponents()
    pipeline.initialize_ar({})
    return pipeline


def make_batch():
    return DiffusionRequestBatch(
        requests=[
            OmniDiffusionRequest(
                prompt="A forest",
                sampling_params=OmniDiffusionSamplingParams(num_frames=5, height=32, width=32, seed=7),
                request_id="hy-nomas-unit",
            )
        ]
    )


def test_forward_preserves_display_range_and_sampling_arguments(pipeline):
    result = pipeline(make_batch())
    result.media.validate()
    assert result.media.video.spec.value_range is VideoValueRange.ZERO_TO_ONE
    noise = torch.randn((1, 2, 2, 1, 1), generator=torch.Generator().manual_seed(7))
    latents = noise.clone()
    for _ in range(8):
        latents -= 0.25 / 8
    expected = (pipeline.decode(latents).sample[:, :, :5] * 0.5 + 0.5).clamp(0, 1)
    torch.testing.assert_close(result.media.video.tensor, expected, rtol=0, atol=0)
    assert len(pipeline.calls) == 9 and pipeline.calls[-1]["commit"]
    assert pipeline.encoded == ["A forest"]
    assert pipeline._offline_session is None


def test_failure_releases_cache_and_allows_next_request(pipeline):
    pipeline.fail_clean = True
    with pytest.raises(RuntimeError, match="injected clean-forward failure"):
        pipeline(make_batch())
    assert pipeline._offline_session is None
    assert not pipeline._ar_sessions
    pipeline.fail_clean = False
    pipeline(make_batch()).media.validate()


def test_output_shape_is_checked(pipeline):
    pipeline.vae.decode = lambda latents: SimpleNamespace(sample=torch.zeros(1, 3, 4, 32, 32))
    with pytest.raises(RuntimeError, match="expected"):
        pipeline(make_batch())


def test_sp_rejects_uneven_tokens_before_inference(pipeline):
    pipeline.sp_size = 2
    with pytest.raises(OmniClientError, match="spatial tokens per frame"):
        pipeline(make_batch())
    assert not pipeline.calls


@pytest.mark.parametrize("fails", [False, True])
def test_reference_sdpa_policy_is_scoped_to_inference(pipeline, fails, monkeypatch):
    original = pipeline.transformer.forward_with_cache

    def forward_with_cache(**kwargs):
        assert torch.backends.cuda.cudnn_sdp_enabled()
        if fails:
            raise RuntimeError("injected failure")
        return original(**kwargs)

    monkeypatch.setattr(pipeline, "_component_context", HyNomasPipeline._component_context.__get__(pipeline))
    pipeline.transformer.forward_with_cache = forward_with_cache
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        if fails:
            with pytest.raises(RuntimeError, match="injected failure"):
                pipeline(make_batch())
        else:
            pipeline(make_batch())
        assert not torch.backends.cuda.cudnn_sdp_enabled()


@pytest.mark.parametrize(
    "components,expected_cpu_moves",
    [
        (None, (("transformer", "text_encoder"), ("text_encoder", "vae", "transformer"), ("transformer", "vae"))),
        (frozenset({"dit"}), (("transformer",), ("transformer",), ("transformer",))),
        (frozenset({"text_encoder"}), (("text_encoder",), ("text_encoder",), ())),
        (
            frozenset({"dit", "text_encoder"}),
            (("transformer", "text_encoder"), ("text_encoder", "transformer"), ("transformer",)),
        ),
    ],
)
def test_module_offload_selects_components_and_removes_hooks(monkeypatch, components, expected_cpu_moves):
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Linear(2, 2)
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    pipeline._cpu_offload = False
    pipeline._resident_component_id = None
    names = {id(getattr(pipeline, name)): name for name in ("transformer", "text_encoder", "vae")}
    moves = []

    def move_params(module, target_device, **kwargs):
        moves.append((names[id(module)], target_device.type))
        return False

    monkeypatch.setattr(sequential_backend.SequentialOffloadHook, "_move_params", staticmethod(move_params))
    monkeypatch.setattr(sequential_backend.current_omni_platform, "synchronize", lambda: None)
    monkeypatch.setattr(sequential_backend.current_omni_platform, "get_free_memory", lambda: 0)
    pipeline.enable_omni_model_cpu_offload(
        device=torch.device("cuda:0"), pin_memory=False, offload_components=components
    )
    assert pipeline._cpu_offload
    for name, expected in zip(("text_encoder", "transformer", "vae"), expected_cpu_moves):
        moves.clear()
        with pipeline._component_context(getattr(pipeline, name)):
            assert pipeline._resident_component_id == id(getattr(pipeline, name))
        assert [module for module, device in moves if device == "cpu"] == list(expected)
        assert [module for module, device in moves if device == "cuda"] == [name]
        assert pipeline._resident_component_id is None
    pipeline.disable_omni_model_cpu_offload()
    assert not pipeline._cpu_offload
    for module in (pipeline.transformer, pipeline.text_encoder, pipeline.vae):
        assert module._hook_registry.get_hook("sequential_offload") is None


@pytest.mark.parametrize("components", [frozenset(), frozenset({"vae"}), frozenset({"dit", "vae"})])
def test_module_offload_rejects_unsupported_components(components, mocker):
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Linear(2, 2)
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    pipeline._cpu_offload = False
    enable = mocker.patch.object(pipeline_hy_nomas, "apply_sequential_offload")
    with pytest.raises(ValueError, match="offload components must not be empty|Unknown diffusion offload component"):
        pipeline.enable_omni_model_cpu_offload(device=torch.device("cuda:0"), offload_components=components)
    enable.assert_not_called()
    assert not pipeline._cpu_offload


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"parallel_config": DiffusionParallelConfig(tensor_parallel_size=2)}, "only supports Ulysses"),
        ({"parallel_config": DiffusionParallelConfig(ring_degree=2)}, "only supports Ulysses"),
        ({"parallel_config": DiffusionParallelConfig(allgather_degree=2)}, "only supports Ulysses"),
        ({"parallel_config": DiffusionParallelConfig(ulysses_degree=3)}, "16 attention heads"),
        ({"parallel_config": DiffusionParallelConfig(ulysses_mode="advanced_uaa")}, "strict Ulysses"),
        ({"parallel_config": DiffusionParallelConfig(ulysses_a2a_permute=True)}, "strict Ulysses"),
        ({"parallel_config": DiffusionParallelConfig(vae_patch_parallel_size=2)}, "VAE"),
        ({"dtype": torch.float32}, "bfloat16"),
        ({"cache_backend": "cache_dit"}, "cache backend"),
        ({"quantization_config": "int8"}, "unquantized"),
        ({"lora_path": "adapter"}, "LoRA"),
        ({"custom_pipeline_args": {"arbitrary": True}}, "custom_pipeline_args"),
        ({"enable_layerwise_offload": True}, "layer offload"),
        ({"flow_shift": 3.0}, "flow_shift=5"),
        ({"enforce_eager": False, "diffusion_compile_granularity": "full"}, "full transformer compilation"),
    ],
)
def test_unsupported_engine_features(overrides, match):
    config = SimpleNamespace(
        distributed_executor_backend="mp",
        model_class_name=None,
        num_gpus=1,
        parallel_config=DiffusionParallelConfig(data_parallel_size=1),
        dtype=torch.bfloat16,
        cache_backend="none",
        quantization_config=None,
        lora_path=None,
        custom_pipeline_args={},
        flow_shift=None,
        enforce_eager=True,
        diffusion_compile_granularity="regional",
        step_execution=False,
        streaming_output=False,
    )
    config.__dict__.update(overrides)
    with pytest.raises(ValueError, match=match):
        validate_config(config)


@pytest.mark.parametrize("sp_size", [1, 2, 4, 8, 16])
def test_supported_ulysses_config(sp_size):
    validate_config(
        OmniDiffusionConfig(
            distributed_executor_backend="mp",
            parallel_config=DiffusionParallelConfig(ulysses_degree=sp_size),
            dtype=torch.bfloat16,
        )
    )


class CustomUniExecutor(UniProcDiffusionExecutor):
    pass


@pytest.mark.parametrize(
    "backend",
    [
        None,
        "uni",
        UniProcDiffusionExecutor,
        CustomUniExecutor,
        "vllm_omni.diffusion.executor.uniproc_executor.UniProcDiffusionExecutor",
    ],
)
def test_model_rejects_in_process_execution_before_initializing_components(monkeypatch, backend):
    device = Mock(side_effect=AssertionError("Component initialization must not start"))
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", device)
    config = OmniDiffusionConfig(dtype=torch.bfloat16, distributed_executor_backend=backend)
    with pytest.raises(ValueError, match="requires process isolation"):
        HyNomasPipeline(od_config=config)
    device.assert_not_called()


@pytest.mark.parametrize("backend", ["mp", "ray"])
def test_model_accepts_isolated_executor(backend):
    validate_config(OmniDiffusionConfig(dtype=torch.bfloat16, distributed_executor_backend=backend))


def make_transformer():
    transformer = nn.Module()
    transformer.base = nn.Linear(2, 2, bias=False)
    transformer.action_modules = nn.ModuleList([nn.Linear(2, 2, bias=False)])
    return transformer


@pytest.mark.parametrize("camera_name", ["action_modules.0.weight", "double_blocks.0.action_module.weight"])
def test_complete_dit_checkpoint_with_reference_aliases(tmp_path, camera_name):
    model = make_transformer()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"base.weight": model.base.weight, camera_name: model.action_modules[0].weight}, str(checkpoint))
    expected = {name: value.clone() for name, value in model.state_dict().items()}
    for parameter in model.parameters():
        parameter.data.zero_()
    assert load_transformer_weights(model, load_file(str(checkpoint)).items()) == set(expected)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


def test_missing_camera_weights_are_rejected(tmp_path):
    model = make_transformer()
    checkpoint = tmp_path / "model.safetensors"
    save_file({"base.weight": model.base.weight}, str(checkpoint))
    with pytest.raises(ValueError, match="action_modules.0.weight"):
        load_transformer_weights(model, load_file(str(checkpoint)).items())


def test_standard_loader_accounts_for_all_components():
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = make_transformer()
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    weights = [
        (f"{component}.{name}", tensor.clone())
        for component in ("transformer", "vae")
        for name, tensor in getattr(pipeline, component).state_dict().items()
    ]
    loaded = pipeline.load_weights(weights)
    assert loaded == set(dict(pipeline.named_parameters()))
    with pytest.raises(ValueError, match="Unexpected Hy-Nomas component"):
        pipeline.load_weights(iter((("weight", torch.zeros(1)),)))
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_weights(iter(()))


def test_standard_loader_reads_shards_and_remaps_diffusers_parameters(checkpoint):
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Module()
    pipeline.transformer.double_blocks = nn.ModuleList([nn.ModuleDict({"img_attn_q": nn.Linear(2, 2, bias=False)})])
    pipeline.transformer.txt_in = nn.Module()
    pipeline.transformer.txt_in.individual_token_refiner = nn.Module()
    pipeline.transformer.txt_in.individual_token_refiner.blocks = nn.ModuleList(
        [nn.ModuleDict({"self_attn_qkv": nn.Linear(2, 2, bias=False)})]
    )
    pipeline.vae = nn.ModuleDict({"decoder": nn.Linear(2, 2, bias=False)})
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.to(dtype=torch.bfloat16)
    expected = {name: value.clone() for name, value in pipeline.named_parameters()}
    components = {
        "transformer": {
            "double_blocks.0.attn.img_attn_q.weight": expected["transformer.double_blocks.0.img_attn_q.weight"],
            "txt_in.individual_token_refiner.blocks.0.attn.self_attn_qkv.weight": expected[
                "transformer.txt_in.individual_token_refiner.blocks.0.self_attn_qkv.weight"
            ],
        },
        "decoder": {"decoder.weight": expected["vae.decoder.weight"]},
    }
    for component, tensors in components.items():
        weight_map = {}
        for index, (name, value) in enumerate(tensors.items()):
            filename = f"diffusion_pytorch_model-{index + 1:05d}-of-{len(tensors):05d}.safetensors"
            save_file({name: value}, str(checkpoint / component / filename))
            weight_map[name] = filename
        (checkpoint / component / "diffusion_pytorch_model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": weight_map})
        )
    for module in (pipeline.transformer, pipeline.vae):
        for parameter in module.parameters():
            parameter.data.zero_()
    pipeline.weights_sources = [
        DiffusersPipelineLoader.ComponentSource(str(checkpoint), component, None, prefix=prefix, fall_back_to_pt=False)
        for component, prefix in (("transformer", "transformer."), ("decoder", "vae."))
    ]
    loader = DiffusersPipelineLoader(LoadConfig(), OmniDiffusionConfig(model=str(checkpoint), dtype=torch.bfloat16))
    assert pipeline.load_weights(loader.get_all_weights(pipeline)) == set(expected)
    for name, value in pipeline.named_parameters():
        torch.testing.assert_close(value, expected[name], rtol=0, atol=0)


@pytest.mark.parametrize("kind", ["duplicate", "unknown", "shape"])
def test_malformed_dit_weight_stream_is_rejected(kind):
    transformer = make_transformer()
    weights = list(transformer.state_dict().items())
    if kind == "duplicate":
        weights.append(("double_blocks.0.action_module.weight", weights[1][1]))
    elif kind == "unknown":
        weights.append(("extra.weight", torch.zeros(2, 2)))
    else:
        weights[0] = (weights[0][0], torch.zeros(3, 3))
    with pytest.raises((ValueError, AssertionError)):
        load_transformer_weights(transformer, weights)


class TokenizerStub:
    prompt_template_video = {"crop_start": 3}

    def text2tokens(self, prompts, *, max_length, **kwargs):
        ids = [list(text.encode())[:max_length] for text in prompts]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.ones(1, len(ids[0]))}


@pytest.mark.local_model
@pytest.mark.parametrize("structured", ["caption", "empty", "list", "mapping", "single", "overlap", "missing_time"])
def test_action_masks_match_reference_helpers(structured):
    reference = pytest.importorskip("hymm.pipeline.base")
    model = reference.BasePipeline.__new__(reference.BasePipeline)
    nn.Module.__init__(model)
    model.args = OmegaConf.create({"text_len": 1280, "inference": {"video_fps": 24}})
    model.text_encoder = TokenizerStub()
    model.t_ratio = 4
    actions: list[dict[str, str | float]] = [
        {"start_time": "00:00.5", "end_time": "00:01.0", "action_description": "walk"},
        {"start_time": "00:01.0", "end_time": "00:01.5", "action_description": "turn"},
    ]
    if structured == "single":
        actions = [{"start_time": 0, "end_time": 10, "action_description": "walk"}]
    elif structured == "overlap":
        actions[1]["start_time"] = 0.5
    elif structured == "missing_time":
        del actions[0]["start_time"]
    contained_actions = (
        {f"Action{index}": action for index, action in enumerate(actions)} if structured == "mapping" else actions
    )
    prompt = json.dumps({"segments": [{"description": "A forest", "contained_actions": contained_actions}]})
    if structured == "caption":
        prompt = "A forest beside a river."
    elif structured == "empty":
        prompt = json.dumps({"segments": []})
    if structured == "overlap":
        with pytest.raises(ValueError, match="Overlapping"):
            model._compute_action_video_windows(prompt, 0, 12)
        with pytest.raises(ValueError, match="overlap"):
            action_conditioning(prompt, model.text_encoder, 1280, 12)
        return
    start = model._compute_action_start(prompt)
    offsets = model._compute_action_text_offsets(prompt)
    windows = model._compute_action_video_windows(prompt, 0, 12)
    expected = (start, offsets, windows) if offsets is not None and windows is not None else (start, None, None)
    assert action_conditioning(prompt, model.text_encoder, 1280, 12) == expected


@pytest.mark.local_model
def test_reference_transformer_factory_places_every_parameter(checkpoint):
    pytest.importorskip("hymm.models.dit.modules")
    config = build_reference_config(checkpoint, str(checkpoint / "text_encoder"))
    transformer = build_transformer(config, torch.device("meta"), TRANSFORMER_CONFIG)
    parameters = dict(transformer.named_parameters())
    assert len(parameters) == 2268
    assert all(parameter.device.type == "meta" for parameter in parameters.values())
    assert all(parameter.dtype == torch.bfloat16 for parameter in parameters.values())


@pytest.mark.local_model
def test_decoder_factory_places_every_parameter():
    pytest.importorskip("hymm.models.vae.pixelumm_decoder.autoencoder")
    decoder = build_decoder(DECODER_CONFIG, torch.device("meta"))
    parameters = dict(decoder.named_parameters())
    assert len(parameters) == 476
    assert all(name.startswith("decoder.") for name in parameters)
    assert all(
        parameter.device.type == "meta" and parameter.dtype == torch.bfloat16 for parameter in parameters.values()
    )


def _check_reference_parallelism(rank, rendezvous):
    from hymm.distributed.parallel_states import get_sequence_parallel_state, nccl_info

    from vllm_omni.diffusion.models.hy_nomas import hy_nomas_transformer

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2, timeout=timedelta(seconds=60))
    try:
        with pytest.MonkeyPatch.context() as patch:
            patch.setenv("RANK", "0")
            group = Mock(ulysses_group=dist.group.WORLD, ulysses_world_size=2, world_size=2)
            patch.setattr(hy_nomas_transformer, "get_sp_group", lambda: group)
            mesh = hy_nomas_transformer.initialize_reference_parallelism(2, torch.device("cpu"))
            assert mesh.get_group("sp") is dist.group.WORLD
            assert nccl_info.sp_group is dist.group.WORLD
            assert nccl_info.sp_size == 2
            assert nccl_info.rank_within_spgroup == nccl_info.global_rank == rank
            assert nccl_info.parallel_dims.sp_group is dist.group.WORLD
            assert get_sequence_parallel_state()
            value = torch.tensor(rank + 1)
            dist.all_reduce(value, group=nccl_info.sp_group)
            assert value.item() == 3

            assert hy_nomas_transformer.initialize_reference_parallelism(1, torch.device("cpu")) is None
            assert not get_sequence_parallel_state()
            assert nccl_info.sp_size == 1
            assert nccl_info.sp_group is None
            assert nccl_info.device_mesh is None
            assert dist.is_initialized()

            group.ulysses_world_size = 4
            with pytest.raises(ValueError, match="match ulysses_degree"):
                hy_nomas_transformer.initialize_reference_parallelism(2, torch.device("cpu"))
    finally:
        dist.destroy_process_group()


@pytest.mark.local_model
def test_reference_reuses_process_group_and_resets_single_device(tmp_path):
    pytest.importorskip("hymm.distributed.parallel_states", reason="Install the pinned Hy-Nomas reference checkout")
    mp.spawn(_check_reference_parallelism, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)
