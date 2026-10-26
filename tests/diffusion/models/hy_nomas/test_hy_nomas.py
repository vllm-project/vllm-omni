# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
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

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.executor.uniproc_executor import UniProcDiffusionExecutor
from vllm_omni.diffusion.media import VideoValueRange
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.hy_nomas import pipeline_hy_nomas
from vllm_omni.diffusion.models.hy_nomas.configuration_hy_nomas import DIT_FILE, REQUIRED_FILES
from vllm_omni.diffusion.models.hy_nomas.hy_nomas_transformer import build_transformer, load_transformer_weights
from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import (
    HyNomasPipeline,
    build_reference_config,
    validate_config,
)
from vllm_omni.diffusion.models.hy_nomas.processing_hy_nomas import action_conditioning, normalize_request
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
    transformer = nn.Linear(2, 2, bias=False)
    transformer.patch_size = (1, 1, 1)
    transformer.config = SimpleNamespace(in_channels=2)
    transformer.double_blocks = []
    transformer.heads_num = 1
    transformer.hidden_size = 2
    save_file(transformer.state_dict(), str(tmp_path / DIT_FILE))
    text_encoder = nn.Linear(2, 2)
    vae = nn.Linear(2, 2)
    vae.spatial_compression_ratio = 32
    loader = ModuleType("hymm.models.checkpointing.load")
    monkeypatch.setattr(loader, "load_text_encoder", Mock(return_value=text_encoder), raising=False)
    decoder = ModuleType("hymm.models.vae.pixelumm_decoder")
    monkeypatch.setattr(decoder, "maybe_load_pixelumm_decoder", Mock(return_value=vae), raising=False)
    monkeypatch.setitem(sys.modules, loader.__name__, loader)
    monkeypatch.setitem(sys.modules, decoder.__name__, decoder)
    monkeypatch.setattr(pipeline_hy_nomas, "build_transformer", Mock(return_value=transformer))
    monkeypatch.setattr(pipeline_hy_nomas, "get_local_device", lambda: torch.device("cuda:0"))
    monkeypatch.setattr(pipeline_hy_nomas, "initialize_reference_parallelism", Mock(return_value=None))
    monkeypatch.setattr(pipeline_hy_nomas, "download_weights_from_hf_specific", download)
    monkeypatch.setattr(
        pipeline_hy_nomas,
        "build_reference_config",
        Mock(return_value=DictConfig({"inference": {"cache_noise_sigma": 0.206897}})),
    )
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_OFFLINE", False)
    config = OmniDiffusionConfig(
        model="tencent/Hy-Nomas", revision="test-revision", dtype=torch.bfloat16, distributed_executor_backend="mp"
    )
    pipeline = HyNomasPipeline(od_config=config)
    assert pipeline.transformer is transformer and pipeline.text_encoder is text_encoder and pipeline.vae is vae
    [source] = pipeline.weights_sources
    assert source.subfolder == "DiT/distilled" and source.prefix == "transformer."
    assert source.allow_patterns_overrides == [Path(DIT_FILE).name]
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


@pytest.mark.parametrize("components", [None, frozenset({"dit"}), frozenset({"text_encoder"})])
def test_module_offload_selects_components_and_removes_hooks(monkeypatch, components):
    pipeline = HyNomasPipeline.__new__(HyNomasPipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = nn.Linear(2, 2)
    pipeline.text_encoder = nn.Linear(2, 2)
    pipeline.vae = nn.Linear(2, 2)
    pipeline._cpu_offload = False
    enable = Mock()
    disable = Mock()
    monkeypatch.setattr(pipeline_hy_nomas, "apply_sequential_offload", enable)
    monkeypatch.setattr(pipeline_hy_nomas, "remove_sequential_offload", disable)
    pipeline.enable_omni_model_cpu_offload(device=torch.device("cuda:0"), offload_components=components)
    selected = enable.call_args.kwargs
    assert selected["offload_dit_modules"] == (
        [pipeline.transformer] if components is None or "dit" in components else []
    )
    expected_stages = (
        [pipeline.text_encoder, pipeline.vae]
        if components is None
        else ([pipeline.text_encoder] if "text_encoder" in components else [])
    )
    assert selected["offload_encoder_modules"] == expected_stages
    assert pipeline._cpu_offload
    pipeline.disable_omni_model_cpu_offload()
    disable.assert_called_once_with([pipeline.transformer, pipeline.text_encoder, pipeline.vae])
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
    weights = [(f"transformer.{name}", tensor.clone()) for name, tensor in pipeline.transformer.state_dict().items()]
    loaded = pipeline.load_weights(weights)
    assert loaded == set(dict(pipeline.named_parameters()))
    with pytest.raises(ValueError, match="Unexpected Hy-Nomas component"):
        pipeline.load_weights(iter((("weight", torch.zeros(1)),)))
    with pytest.raises(ValueError, match="missing"):
        pipeline.load_weights(iter(()))


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
def test_reference_transformer_factory_places_every_parameter(tmp_path):
    pytest.importorskip("hymm.models.dit.modules")
    config = build_reference_config(tmp_path, "Qwen/Qwen2.5-VL-7B-Instruct")
    transformer = build_transformer(config, torch.device("meta"))
    parameters = dict(transformer.named_parameters())
    assert len(parameters) == 2268
    assert all(parameter.device.type == "meta" for parameter in parameters.values())
    assert all(parameter.dtype == torch.bfloat16 for parameter in parameters.values())


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
