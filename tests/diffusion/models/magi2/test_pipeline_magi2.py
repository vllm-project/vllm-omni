# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

from dataclasses import dataclass, field
from types import MethodType
from typing import Any
from unittest.mock import Mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from PIL import Image

from vllm_omni.diffusion.data import OmniDiffusionConfig, resolve_model_class_name
from vllm_omni.diffusion.model_metadata import get_diffusion_model_metadata
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import (
    MAGI2_AUDIO_SAMPLE_RATE,
    MAGI2_MODEL_REVISION,
    Magi2Pipeline,
    _Magi2StagedComponent,
    _resolve_checkpoint_root,
    _validate_native_topology,
)
from vllm_omni.diffusion.models.magi2.preview_data_proxy import Magi2DataProxy
from vllm_omni.diffusion.models.magi2.sampler_magi2 import CFGConfig, Magi2PreviewSampler
from vllm_omni.diffusion.registry import DiffusionModelRegistry
from vllm_omni.diffusion.utils.hf_utils import is_diffusion_model
from vllm_omni.errors import OmniClientError
from vllm_omni.model_extras.registry import (
    get_extra_body_params,
    should_preserve_reference_image_size,
)

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


@dataclass
class _SamplingStub:
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_rate: float | None = None
    resolved_frame_rate: float | None = None
    num_frames: int = 1
    num_inference_steps: int | None = None
    num_outputs_per_prompt: int = 1
    seed: int | None = 42
    generator: torch.Generator | list[torch.Generator] | None = None
    extra_args: dict[str, Any] = field(default_factory=dict)


@dataclass
class _RequestStub:
    prompts: list[Any]
    sampling_params: _SamplingStub
    num_reqs: int = 1


@dataclass
class _ParallelStub:
    pipeline_parallel_size: int = 1
    data_parallel_size: int = 1
    tensor_parallel_size: int = 1
    sequence_parallel_size: int = 1
    ulysses_degree: int = 1
    ring_degree: int = 1
    allgather_degree: int = 1
    cfg_parallel_size: int = 1
    vae_patch_parallel_size: int = 1
    text_encoder_tp_size: int = 1
    enable_expert_parallel: bool = False
    use_hsdp: bool = False


@dataclass
class _TopologyStub:
    parallel_config: _ParallelStub = field(default_factory=_ParallelStub)
    enable_cpu_offload: bool = False
    enable_layerwise_offload: bool = False
    enable_distributed_layerwise_offload: bool = False
    dlo_use_allgather: bool = True
    quantization_config: object = None
    cache_backend: str = "none"
    custom_pipeline_args: dict[str, object] = field(default_factory=dict)
    additional_config: dict[str, object] = field(default_factory=lambda: {"magi2_allow_unsupported_topology": True})


class _FakeNativeRuntime:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def evaluate(self, **kwargs):
        self.calls.append(kwargs)
        video = np.zeros((2, 8, 12, 3), dtype=np.uint8)
        audio = np.zeros((32, 2), dtype=np.float32)
        return video, audio


def _pipeline() -> tuple[Magi2Pipeline, _FakeNativeRuntime]:
    pipe = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipe)
    pipe.deterministic = False
    pipe.enable_diffusion_pipeline_profiler = False
    runtime = _FakeNativeRuntime()

    def evaluate(_self, **kwargs):
        return runtime.evaluate(**kwargs)

    pipe._evaluate_preview = MethodType(evaluate, pipe)
    return pipe, runtime


def _request(prompt, **sampling_overrides):
    sampling = _SamplingStub()
    for key, value in sampling_overrides.items():
        setattr(sampling, key, value)
    return _RequestStub(prompts=[prompt], sampling_params=sampling)


def _checkpoint_tree(root) -> None:
    files = (
        "preview/model.safetensors.index.json",
        "text_encoder/config.json",
        "text_encoder/model.safetensors.index.json",
        "vae/Wan2.2_VAE.pth",
        "turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json",
        "turbo_vae/checkpoint.ckpt",
        "stable-audio-open-1.0/model_config.json",
        "stable-audio-open-1.0/model.safetensors",
    )
    for relative in files:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_official_id_detection_and_metadata():
    assert is_diffusion_model("sand-ai/MAGI-2-preview")
    assert is_diffusion_model("https://huggingface.co/sand-ai/MAGI-2-preview")
    assert resolve_model_class_name("sand-ai/MAGI-2-preview") == "Magi2Pipeline"
    assert DiffusionModelRegistry._try_load_model_cls("Magi2Pipeline") is Magi2Pipeline
    metadata = get_diffusion_model_metadata("Magi2Pipeline")
    assert metadata.supports_multimodal_inputs
    assert metadata.max_multimodal_image_inputs == 1
    assert should_preserve_reference_image_size(
        "Magi2Pipeline",
        model="sand-ai/MAGI-2-preview",
    )
    assert {"seconds", "resolution"} <= get_extra_body_params("Magi2Pipeline")


def test_local_preview_signature_detection_without_refiner(tmp_path):
    _checkpoint_tree(tmp_path)
    assert is_diffusion_model(str(tmp_path))
    assert resolve_model_class_name(str(tmp_path)) == "Magi2Pipeline"
    config = OmniDiffusionConfig(model=str(tmp_path))
    config.enrich_config()
    assert config.model_class_name == "Magi2Pipeline"
    assert config.supports_multimodal_inputs


def test_huggingface_url_resolves_pinned_snapshot(tmp_path, monkeypatch):
    _checkpoint_tree(tmp_path)
    api = Mock()
    api.snapshot_download.side_effect = lambda *, repo_id, revision: (
        str(tmp_path)
        if (repo_id, revision) == ("sand-ai/MAGI-2-preview", MAGI2_MODEL_REVISION)
        else pytest.fail("unexpected snapshot request")
    )
    monkeypatch.setattr(
        "vllm.transformers_utils.repo_utils.hf_api",
        lambda: api,
    )
    assert _resolve_checkpoint_root(
        "https://huggingface.co/sand-ai/MAGI-2-preview",
        None,
    ) == str(tmp_path.resolve())


def test_forward_uses_native_540p_preview_defaults(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )
    result = pipe(_request("A fox walks through snow"))

    call = runtime.calls[0]
    assert call["prompt"] == "A fox walks through snow"
    assert call["image"] is None
    assert (call["width"], call["height"]) == (896, 512)
    assert call["num_inference_steps"] == 100
    assert result.output["metadata"]["video"]["fps"] == 12.5
    assert result.output["metadata"]["audio"]["sample_rate"] == MAGI2_AUDIO_SAMPLE_RATE
    assert result.stage_durations["magi2_preview_e2e"] >= 0


def test_forward_maps_272p_i2v_and_output_resize(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )
    image = Image.new("RGB", (16, 9), "white")
    prompt = {
        "prompt": "The first frame begins moving",
        "multi_modal_data": {"image": image},
    }
    result = pipe(
        _request(
            prompt,
            width=448,
            height=256,
            num_inference_steps=1,
            extra_args={
                "output_width": 6,
                "output_height": 4,
            },
        )
    )

    call = runtime.calls[0]
    assert call["image"] is image
    assert pipe._ensure_figure_token(call["prompt"]) == (
        "The first frame begins moving\nreference_layer:The first frame refers to <Figure 1>"
    )
    assert (call["width"], call["height"]) == (448, 256)
    assert call["num_inference_steps"] == 1
    assert result.output["payload"]["video"].shape == (2, 4, 6, 3)


def test_forward_rejects_conflicting_resolution_and_sampling_geometry(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )

    with pytest.raises(OmniClientError, match="requires 896x512"):
        pipe(
            _request(
                "A fox walks through snow",
                width=448,
                height=256,
                extra_args={"resolution": "540p"},
            )
        )
    assert not runtime.calls


@pytest.mark.parametrize(
    ("prompt", "extra_args", "message"),
    [
        ("", {}, "non-empty prompt"),
        ("prompt", {"seconds": 5}, "10-second clips"),
        ("prompt", {"duration": 5}, "10-second clips"),
        ("prompt", {"resolution": "720p"}, "Unsupported native"),
        ("prompt", {"resolution": "1080p"}, "Unsupported native"),
        ("prompt", {"resolution": "540p", "use_refiner": True}, "refiner"),
        ("prompt", {"output_width": 0, "output_height": 4}, "positive integers"),
    ],
)
def test_forward_rejects_invalid_preview_requests(
    monkeypatch,
    prompt,
    extra_args,
    message,
):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )
    with pytest.raises(OmniClientError, match=message):
        pipe(_request(prompt, extra_args=extra_args))
    assert not runtime.calls


def test_forward_rejects_zero_inference_steps_before_generation(monkeypatch):
    pipe, runtime = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )
    with pytest.raises(OmniClientError, match="inference steps must be positive"):
        pipe(_request("prompt", num_inference_steps=0))
    assert not runtime.calls


def test_forward_stops_peak_monitor_when_initial_synchronize_fails(monkeypatch):
    pipe, runtime = _pipeline()
    pipe.enable_diffusion_pipeline_profiler = True
    monitor = Mock(peak_bytes=0)
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: True,
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock(side_effect=RuntimeError("sync failed")))
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2._PeakReservedMonitor",
        Mock(return_value=monitor),
    )

    with pytest.raises(RuntimeError, match="sync failed"):
        pipe(_request("A fox walks through snow"))

    monitor.start.assert_called_once_with()
    monitor.stop.assert_called_once_with()
    assert not runtime.calls


def test_forward_does_not_start_peak_monitor_without_profiler(monkeypatch):
    pipe, runtime = _pipeline()
    monitor_factory = Mock(side_effect=AssertionError("unexpected memory monitor"))
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_cuda",
        lambda: True,
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: True,
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(torch.accelerator, "synchronize", Mock())
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2._PeakReservedMonitor",
        monitor_factory,
    )

    result = pipe(_request("A fox walks through snow"))

    monitor_factory.assert_not_called()
    assert result.peak_memory_mb == 0.0
    assert len(runtime.calls) == 1


def test_forward_rejects_multiple_images(monkeypatch):
    pipe, _ = _pipeline()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.current_omni_platform.is_available",
        lambda: False,
    )
    prompt = {
        "prompt": "animate",
        "multi_modal_data": {"image": [Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))]},
    }
    with pytest.raises(OmniClientError, match="at most one input image"):
        pipe(_request(prompt))


def test_pipeline_rejects_unknown_initialization_arguments():
    with pytest.raises(TypeError, match=r"Unexpected MAGI-2.*unknown_option"):
        Magi2Pipeline(None, unknown_option=True)


def _topology_config(**overrides):
    parallel = _ParallelStub()
    config = _TopologyStub(parallel_config=parallel)
    for key, value in overrides.items():
        if hasattr(parallel, key):
            setattr(parallel, key, value)
        else:
            setattr(config, key, value)
    return config


def test_native_topology_accepts_qualified_tensor_parallelism():
    _validate_native_topology(_topology_config(tensor_parallel_size=4))


def test_native_topology_rejects_nondivisible_tensor_parallelism():
    config = _topology_config(tensor_parallel_size=3)
    with pytest.raises(ValueError, match="does not divide"):
        _validate_native_topology(config)


def test_native_topology_accepts_single_device_layerwise_with_cpu_staging():
    config = _topology_config(
        enable_layerwise_offload=True,
        additional_config={},
    )
    _validate_native_topology(config)

    config.enable_cpu_offload = True
    _validate_native_topology(config)

    config.enable_layerwise_offload = False
    with pytest.raises(ValueError, match="Combine --enable-cpu-offload"):
        _validate_native_topology(config)


def test_native_topology_requires_dlo_rank_local_mode():
    config = _topology_config(enable_distributed_layerwise_offload=True)
    with pytest.raises(ValueError, match="dlo-no-use-allgather"):
        _validate_native_topology(config)
    config.dlo_use_allgather = False
    _validate_native_topology(config)


def test_native_topology_accepts_four_device_hsdp_cfg_and_vae_layouts():
    _validate_native_topology(
        _topology_config(
            sequence_parallel_size=4,
            ulysses_degree=4,
            use_hsdp=True,
        )
    )
    _validate_native_topology(
        _topology_config(
            sequence_parallel_size=2,
            ulysses_degree=2,
            cfg_parallel_size=2,
        )
    )
    _validate_native_topology(
        _topology_config(
            sequence_parallel_size=4,
            ulysses_degree=4,
            vae_patch_parallel_size=4,
        )
    )


def test_native_topology_rejects_cfg_data_parallel_combination():
    with pytest.raises(ValueError, match="CFG parallelism.*DLO data parallelism"):
        _validate_native_topology(_topology_config(cfg_parallel_size=2, data_parallel_size=2))


def test_cfg_parallel_branch_adapter_preserves_packed_cfg_math():
    sampler = Magi2PreviewSampler(nn.Identity(), Magi2DataProxy(), device="cpu", dtype=torch.float32)
    latent = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 1, 2)
    audio_latent = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    positive_text = torch.ones(1, 3, 4)
    negative_text = torch.full((1, 2, 4), -1.0)
    packed = sampler.prepare_model_input(
        latent=latent,
        audio_latent=audio_latent,
        txt_feat=positive_text,
        null_txt_feat=negative_text,
        t=torch.tensor(500.0),
        cfg_config=CFGConfig(),
    )
    positive, negative = sampler._split_cfg_model_input(packed)

    assert positive.x_t.shape[0] == negative.x_t.shape[0] == 1
    assert packed.ref_video_feat.shape == (2, 2, 0, 1, 2)
    assert packed.ref_video_feat.numel() == 0
    torch.testing.assert_close(positive.txt_feat[:, :3], positive_text)
    torch.testing.assert_close(negative.txt_feat[:, :2], negative_text)

    video_pos = torch.full_like(latent, 3.0)
    video_neg = torch.full_like(latent, 1.0)
    audio_pos = torch.full_like(audio_latent, 4.0)
    audio_neg = torch.full_like(audio_latent, 2.0)
    expected = sampler.cfg_velocity(
        (torch.cat((video_pos, video_neg)), torch.cat((audio_pos, audio_neg))),
        5.0,
        7.0,
        CFGConfig(),
        latent,
        audio_latent,
    )
    actual = sampler.combine_cfg_noise(
        (video_pos, audio_pos),
        (video_neg, audio_neg),
        1.0,
        kwargs={
            "video_txt_guidance_scale": 5.0,
            "audio_txt_guidance_scale": 7.0,
            "cfg_config": CFGConfig(),
            "latent": latent,
            "audio_latent": audio_latent,
        },
    )
    torch.testing.assert_close(actual[0], expected[0])
    torch.testing.assert_close(actual[1], expected[1])


def test_decode_audio_preserves_batch_for_released_preview_shape(monkeypatch):
    class ShapeCheckingDecoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_shape: tuple[int, ...] | None = None

        def decode(self, latent: torch.Tensor) -> torch.Tensor:
            self.seen_shape = tuple(latent.shape)
            if self.seen_shape != (1, 64, 250):
                raise AssertionError(f"unexpected decoder input shape {self.seen_shape}")
            return torch.zeros(1, 2, 512_000)

    decoder = ShapeCheckingDecoder()
    stager = Mock()
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        Mock(return_value=stager),
    )
    component = _Magi2StagedComponent(
        decoder,
        torch.device("cpu"),
        pin_memory=False,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._is_output_rank = True
    pipeline._offload_aux_after_use = True
    pipeline.device_str = "cpu"
    pipeline.audio_decoder = component

    def fake_resample(audio: np.ndarray, target_length: int) -> np.ndarray:
        assert audio.shape == (512_000, 2)
        assert target_length == 441_000
        return np.zeros((target_length, 2), dtype=audio.dtype)

    monkeypatch.setattr("scipy.signal.resample", fake_resample)
    audio = pipeline._decode_audio(torch.zeros(1, 250, 64))

    assert decoder.seen_shape == (1, 64, 250)
    assert audio is not None
    assert audio.shape == (441_000, 2)
    stager.load.assert_called_once_with()
    stager.offload.assert_called_once_with()


class _TinyMagiTransformer(nn.Module):
    _layerwise_offload_blocks_attrs = ["block"]

    def __init__(self) -> None:
        super().__init__()
        self.block = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])


class _NoEagerMoveComponent(_Magi2StagedComponent):
    def to(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("staged MAGI-2 component was moved eagerly")


def _offload_pipeline(monkeypatch):
    stagers = [Mock() for _ in range(4)]
    stager_factory = Mock(side_effect=stagers)
    monkeypatch.setattr(
        "vllm_omni.platforms.current_omni_platform.Stream",
        Mock(return_value=Mock()),
    )
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.magi2.pipeline_magi2.PinnedModuleStager",
        stager_factory,
    )
    pipeline = object.__new__(Magi2Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.transformer = _TinyMagiTransformer()
    components = [
        _NoEagerMoveComponent(
            nn.Linear(2, 2),
            torch.device("cuda:0"),
            pin_memory=False,
        )
        for _ in range(4)
    ]
    (
        pipeline.text_encoder,
        pipeline.image_vae,
        pipeline.video_decoder,
        pipeline.audio_decoder,
    ) = components
    return pipeline, stagers


class _FakeLayerwiseHook:
    def __init__(self) -> None:
        self._prev_hook = None
        self.current_slot = 0
        self.prefetch_layer = Mock()
        self.get_weights = Mock()


def test_dlo_no_allgather_enable_preserves_staged_aux_and_streams_blocks(
    monkeypatch,
):
    from vllm_omni.diffusion.offloader.base import (
        OffloadConfig,
        OffloadStrategy,
    )
    from vllm_omni.diffusion.offloader.distributed_layerwise_backend import (
        DistributedLayerwiseOffloadBackend,
    )

    pipeline, stagers = _offload_pipeline(monkeypatch)
    hooks: list[_FakeLayerwiseHook] = []

    def fake_apply(*args, **kwargs):
        del args, kwargs
        hook = _FakeLayerwiseHook()
        hooks.append(hook)
        return hook

    monkeypatch.setattr(
        "vllm_omni.diffusion.offloader.distributed_layerwise_backend.apply_distributed_block_hook",
        fake_apply,
    )
    backend = DistributedLayerwiseOffloadBackend(
        OffloadConfig(
            strategy=OffloadStrategy.DISTRIBUTED_LAYER_WISE,
            pin_cpu_memory=False,
            dlo_use_allgather=False,
        ),
        torch.device("cpu"),
    )
    backend._allocate_shared_buffers = Mock(return_value=[{}, {}])
    backend._cleanup_after_loading = Mock()
    backend._release_mmap_handles = Mock()

    backend.enable(pipeline)

    assert backend.enabled
    assert backend.dp_size == 1
    assert len(hooks) == len(pipeline.transformer.block)
    assert all(stager.offload.call_count == 1 for stager in stagers)
    assert all(stager.load.call_count == 0 for stager in stagers)
    hooks[-1].prefetch_layer.assert_called_once_with(
        slot=hooks[0].current_slot,
        non_blocking=False,
    )
