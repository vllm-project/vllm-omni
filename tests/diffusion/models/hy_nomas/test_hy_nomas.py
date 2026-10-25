# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import DictConfig
from safetensors.torch import save_file
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.media import VideoValueRange
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import DIT_FILE, REQUIRED_FILES
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline, validate_config
from vllm_omni.diffusion.models.hy_nomas.request import normalize_request
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.utils.hf_utils import (
    _looks_like_hy_nomas,
    is_diffusion_model,
    resolve_native_diffusion_model_class,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_extras import get_extra_body_params, get_video_generation_defaults

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def test_discovery_without_model_index(tmp_path):
    assert _looks_like_hy_nomas("tencent/Hy-Nomas")
    assert _looks_like_hy_nomas(" TENCENT/Hy-Nomas/ ")
    assert resolve_native_diffusion_model_class("tencent/Hy-Nomas") == "HyNomasPipeline"
    assert is_diffusion_model("tencent/Hy-Nomas")
    assert resolve_native_diffusion_model_class(str(tmp_path)) is None
    for relative in REQUIRED_FILES:
        assert not _looks_like_hy_nomas(str(tmp_path))
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    assert resolve_native_diffusion_model_class(str(tmp_path)) == "HyNomasPipeline"
    assert resolve_native_diffusion_model_class("other/Hy-Nomas") is None
    assert get_diffusion_model_metadata("HyNomasPipeline").final_output_type == "video"


def test_native_discovery_does_not_import_runtime_registry():
    code = (
        "import sys; "
        "from vllm_omni.diffusion.utils.hf_utils import resolve_native_diffusion_model_class; "
        "assert resolve_native_diffusion_model_class('tencent/Hy-Nomas') == 'HyNomasPipeline'; "
        "assert resolve_native_diffusion_model_class('other/model') is None; "
        "assert 'vllm_omni.diffusion.registry' not in sys.modules; "
        "assert 'hymm' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("modelscope", ["false", "true"])
def test_hy_nomas_class_resolution_does_not_probe_remote_root_configs(monkeypatch, modelscope):
    from vllm import envs

    envs.disable_envs_cache()
    monkeypatch.setenv("VLLM_USE_MODELSCOPE", modelscope)
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    probe = Mock(side_effect=AssertionError("Native checkpoint discovery must not fetch root configs"))
    monkeypatch.setattr("vllm_omni.diffusion.utils.hf_utils.get_diffusion_model_index", probe)
    monkeypatch.setattr("vllm.transformers_utils.config.get_hf_file_to_dict", probe)
    assert resolve_model_class_name("tencent/Hy-Nomas") == "HyNomasPipeline"
    probe.assert_not_called()


def test_local_hy_nomas_configuration_uses_existing_native_resolution(tmp_path, monkeypatch):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    hub_api = Mock(side_effect=AssertionError("Local checkpoint configuration must not create a Hub client"))
    monkeypatch.setattr("vllm.transformers_utils.repo_utils.hf_api", hub_api)
    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "HyNomasPipeline"
    hub_api.assert_not_called()


def test_local_pipeline_index_keeps_precedence_over_native_signature(tmp_path):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    (tmp_path / "model_index.json").write_text(
        json.dumps({"_class_name": "QwenImagePipeline", "_diffusers_version": "0.40.0"})
    )
    assert not _looks_like_hy_nomas(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "QwenImagePipeline"
    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "QwenImagePipeline"


def test_public_defaults():
    defaults = get_video_generation_defaults("HyNomasPipeline")
    assert (defaults.height, defaults.width, defaults.num_frames) == (704, 1280, 121)
    assert (defaults.num_inference_steps, defaults.guidance_scale, defaults.fps) == (8, 1.0, 24.0)
    assert "segment_frame_boundaries" in get_extra_body_params("HyNomasPipeline")


def test_pipeline_downloads_only_distilled_weights(tmp_path, monkeypatch):
    for relative in REQUIRED_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    download = Mock(return_value=str(tmp_path))
    reference = nn.Module()
    reference.transformer = nn.Linear(2, 2, bias=False)
    save_file(reference.transformer.state_dict(), str(tmp_path / DIT_FILE))
    runtime = ModuleType("vllm_omni.diffusion.models.hy_nomas.runtime")
    monkeypatch.setattr(runtime, "HyNomasRuntime", Mock(return_value=reference), raising=False)
    monkeypatch.setitem(sys.modules, runtime.__name__, runtime)
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(pipeline_hy_nomas, "initialize_reference_parallelism", Mock(return_value=None))
    monkeypatch.setattr(pipeline_hy_nomas, "download_weights_from_hf_specific", download)
    monkeypatch.setattr(
        pipeline_hy_nomas, "build_reference_config", Mock(return_value=DictConfig({"dit_weight": "dit"}))
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    config = OmniDiffusionConfig(model="tencent/Hy-Nomas", revision="test-revision", dtype=torch.bfloat16)
    HyNomasPipeline(od_config=config)
    download.assert_called_once_with(
        "tencent/Hy-Nomas",
        cache_dir=None,
        allow_patterns=list(REQUIRED_FILES),
        revision="test-revision",
        require_all=True,
    )


@pytest.mark.parametrize("encoder", ["", " ", 42, None])
def test_invalid_encoder_fails_before_download(monkeypatch, encoder):
    download = Mock()
    monkeypatch.setattr(pipeline_hy_nomas, "download_weights_from_hf_specific", download)
    config = OmniDiffusionConfig(dtype=torch.bfloat16, custom_pipeline_args={"text_encoder": encoder})
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


@pytest.fixture
def pipeline(monkeypatch):
    module = ModuleType("hymm.models.dit.modules.attention.range_attention")
    monkeypatch.setattr(module, "clear_range_cache", Mock(), raising=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.initialize_ar({})
    pipeline.pipe = Mock()
    pipeline.sp_size = 1
    pipeline.pipe.inference.return_value = torch.linspace(0, 1, 1 * 3 * 5 * 32 * 32).reshape(1, 3, 5, 32, 32)
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


def test_forward_preserves_display_range_and_passes_reference_arguments(pipeline):
    result = pipeline(make_batch())
    result.media.validate()
    assert result.media.video.spec.value_range is VideoValueRange.ZERO_TO_ONE
    torch.testing.assert_close(result.media.video.tensor, pipeline.pipe.inference.return_value)
    kwargs = pipeline.pipe.inference.call_args.kwargs
    assert kwargs["segment_frame_boundaries"] == [0, 5]
    assert kwargs["preserve_kv_cache"] is False
    assert (kwargs["seed"], kwargs["flow_shift"], kwargs["num_inference_steps"]) == (7, 5.0, 8)
    assert pipeline.pipe.kv_cache is None


def test_failure_releases_cache_and_allows_next_request(pipeline):
    def fail(**kwargs):
        pipeline.pipe.kv_cache = [torch.zeros(4)]
        pipeline.pipe.cache_valid_len = 4
        pipeline.pipe.global_frame_counter = 4
        raise RuntimeError("reference failure")

    pipeline.pipe.inference.side_effect = fail
    with pytest.raises(RuntimeError, match="reference failure"):
        pipeline(make_batch())
    assert pipeline.pipe.kv_cache is None
    assert pipeline.pipe.cache_mask_list is None
    assert pipeline.pipe.cache_valid_len == pipeline.pipe.global_frame_counter == 0
    pipeline.pipe.inference.side_effect = None
    pipeline(make_batch()).media.validate()


def test_output_shape_is_checked(pipeline):
    pipeline.pipe.inference.return_value = torch.zeros(1, 3, 4, 32, 32)
    with pytest.raises(RuntimeError, match="expected"):
        pipeline(make_batch())


def test_sp_rejects_uneven_tokens_before_inference(pipeline):
    pipeline.sp_size = 2
    pipeline.pipe.s_ratio = 32
    pipeline.pipe.patch_size = (1, 1, 1)
    with pytest.raises(OmniClientError, match="spatial tokens per frame"):
        pipeline(make_batch())
    pipeline.pipe.inference.assert_not_called()


@pytest.mark.parametrize("fails", [False, True])
def test_reference_sdpa_policy_is_scoped_to_inference(pipeline, fails):
    def inference(**kwargs):
        assert torch.backends.cuda.cudnn_sdp_enabled()
        if fails:
            raise RuntimeError("injected failure")
        return torch.zeros(1, 3, 5, 32, 32)

    pipeline.pipe.inference.side_effect = inference
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        assert not torch.backends.cuda.cudnn_sdp_enabled()
        if fails:
            with pytest.raises(RuntimeError, match="injected failure"):
                pipeline(make_batch())
        else:
            pipeline(make_batch())
        assert not torch.backends.cuda.cudnn_sdp_enabled()


@pytest.mark.parametrize("components", [None, frozenset({"dit"}), frozenset({"text_encoder"})])
def test_module_offload_selects_components_and_removes_hooks(monkeypatch, components):
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.pipe = nn.Module()
    pipeline.pipe.transformer = nn.Linear(2, 2)
    pipeline.pipe.text_encoder = nn.Linear(2, 2)
    pipeline.pipe.vae = nn.Linear(2, 2)
    pipeline.pipe.omni_cpu_offload = False
    enable = Mock()
    disable = Mock()
    monkeypatch.setattr(pipeline_hy_nomas, "apply_sequential_offload", enable)
    monkeypatch.setattr(pipeline_hy_nomas, "remove_sequential_offload", disable)
    pipeline.enable_omni_model_cpu_offload(device=torch.device("cuda:0"), offload_components=components)
    selected = enable.call_args.kwargs
    assert selected["offload_dit_modules"] == (
        [pipeline.pipe.transformer] if components is None or "dit" in components else []
    )
    expected_stages = (
        [pipeline.pipe.text_encoder, pipeline.pipe.vae]
        if components is None
        else ([pipeline.pipe.text_encoder] if "text_encoder" in components else [])
    )
    assert selected["offload_encoder_modules"] == expected_stages
    assert pipeline.pipe.omni_cpu_offload
    pipeline.disable_omni_model_cpu_offload()
    disable.assert_called_once_with([pipeline.pipe.transformer, pipeline.pipe.text_encoder, pipeline.pipe.vae])
    assert not pipeline.pipe.omni_cpu_offload


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
            parallel_config=DiffusionParallelConfig(ulysses_degree=sp_size),
            dtype=torch.bfloat16,
        )
    )
