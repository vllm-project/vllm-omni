# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from __future__ import annotations

import dataclasses
import functools
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from tests.diffusion.models.magi2.test_pipeline_magi2 import _pipeline, _request, _topology_config
from vllm_omni.diffusion.config import get_current_diffusion_config_or_none
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.magi2 import modeling_magi2, modeling_magi2_refiner
from vllm_omni.diffusion.models.magi2 import pipeline_magi2 as pipeline
from vllm_omni.diffusion.models.magi2.configuration_magi2 import MAGI2_REFINER_CONFIG, Magi2RefinerConfig
from vllm_omni.diffusion.models.magi2.parallel import Magi2ParallelGroup
from vllm_omni.diffusion.models.magi2.pipeline_magi2 import Magi2Pipeline, _Magi2StagedComponent
from vllm_omni.diffusion.models.magi2.sampler_magi2_refiner import (
    build_magi2_refiner_scheduler,
    zero_snr_signal_coefficients,
)
from vllm_omni.errors import OmniClientError

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]

_TEXT_CHANNELS = 16
_TINY_CONFIG = Magi2RefinerConfig(
    num_layers=2,
    hidden_size=128,
    head_dim=32,
    num_query_groups=2,
    multimodal_layers=(0,),
    text_in_channels=_TEXT_CHANNELS,
)


class _FakeRefiner(nn.Module):
    """Stand-in refiner transformer that records how it was built, loaded and called."""

    def __init__(self, config: Magi2RefinerConfig, events: list[Any]) -> None:
        super().__init__()
        self.config = config
        self.events = events
        # Initialized from the global generator, as the real refiner's default initialization is.
        self.weight = nn.Parameter(torch.randn(2))
        self.built_in_inference_mode = torch.is_inference_mode_enabled()
        self.built_under_config = get_current_diffusion_config_or_none()
        self.checkpoints: list[Path] = []
        self.loaded_in_inference_mode: bool | None = None
        self.calls: list[dict[str, Any]] = []

    def load_checkpoint(self, checkpoint_dir) -> set[str]:
        self.checkpoints.append(Path(checkpoint_dir))
        self.loaded_in_inference_mode = torch.is_inference_mode_enabled()
        self.events.append("load_checkpoint")
        return set()

    def compile_regions(self, **compile_kwargs: Any) -> None:
        self.events.append(("compile_regions", compile_kwargs))

    def forward(self, *, video, text, coords, modality, window_plan, split_sizes) -> torch.Tensor:
        self.calls.append({"video": video, "text": text, "window_plan": window_plan})
        del coords, modality, split_sizes
        self.events.append(("refiner", window_plan.text_len))
        offset = text.mean() if text.numel() else 0.0
        return torch.tanh(video) * 0.5 + offset


class _RefinerFactory:
    def __init__(self, events: list[Any], config: Magi2RefinerConfig = _TINY_CONFIG) -> None:
        self.events = events
        self.config = config
        self.requested_configs: list[object] = []
        self.models: list[_FakeRefiner] = []

    def __call__(self, config: object = None) -> _FakeRefiner:
        self.requested_configs.append(config)
        self.events.append("build")
        model = _FakeRefiner(self.config, self.events)
        self.models.append(model)
        return model


class _RecordingStager:
    def __init__(
        self,
        events: list[Any],
        module: nn.Module,
        device: torch.device,
        *,
        pin_memory: bool,
        start_loaded: bool = False,
    ) -> None:
        self.events = events
        self.module = module
        self.device = device
        self.pin_memory = pin_memory
        self.loaded = start_loaded
        events.append(("stager", type(module).__name__))

    def load(self) -> None:
        self.loaded = True
        self.events.append(("stage_in", type(self.module).__name__))

    def offload(self) -> None:
        self.loaded = False
        self.events.append(("stage_out", type(self.module).__name__))


@pytest.fixture
def events() -> list[Any]:
    return []


@pytest.fixture
def stagers(monkeypatch: pytest.MonkeyPatch, events: list[Any]) -> list[_RecordingStager]:
    created: list[_RecordingStager] = []

    def factory(module, device, *, pin_memory, start_loaded=False):
        stager = _RecordingStager(events, module, device, pin_memory=pin_memory, start_loaded=start_loaded)
        created.append(stager)
        return stager

    monkeypatch.setattr(pipeline, "PinnedModuleStager", factory)
    return created


@pytest.fixture
def refiner_factory(monkeypatch: pytest.MonkeyPatch, events: list[Any]) -> _RefinerFactory:
    factory = _RefinerFactory(events)
    monkeypatch.setattr(modeling_magi2_refiner, "Magi2RefinerTransformer", factory)
    return factory


def _write_refiner_index(root: Path) -> Path:
    refiner = root / "refiner"
    refiner.mkdir(parents=True, exist_ok=True)
    (refiner / "model.safetensors.index.json").write_text('{"weight_map": {}}')
    return refiner


def _refined_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    events: list[Any],
    **topology: Any,
):
    monkeypatch.setattr(pipeline.current_omni_platform, "is_available", lambda: False)
    pipe, runtime = _pipeline()
    pipe.od_config = _topology_config(**topology)
    pipe.checkpoint_root = str(root)
    pipe.device_str = "cpu"

    def evaluate(_self, **kwargs):
        events.append("evaluate")
        return runtime.evaluate(**kwargs)

    pipe._evaluate_preview = MethodType(evaluate, pipe)
    return pipe, runtime


def _refined_request(**extra: Any):
    return _request("A fox walks through snow", extra_args={"resolution": "1080p", **extra})


@pytest.mark.parametrize("resolution", ["272p", "540p"])
def test_preview_requests_never_build_the_refiner(monkeypatch, tmp_path, events, stagers, refiner_factory, resolution):
    _write_refiner_index(tmp_path)
    pipe, runtime = _refined_pipeline(monkeypatch, tmp_path, events)

    pipe(_request("A fox walks through snow", extra_args={"resolution": resolution}))
    pipe(_request("A fox walks through snow", extra_args={"resolution": resolution}))

    assert len(runtime.calls) == 2
    assert not refiner_factory.models
    assert not stagers
    assert getattr(pipe, "refiner", None) is None
    assert getattr(pipe, "refiner_sampler", None) is None


@pytest.mark.parametrize("pin_memory", [True, False])
def test_first_refined_request_builds_and_loads_the_refiner_once(
    monkeypatch, tmp_path, events, stagers, refiner_factory, pin_memory
):
    refiner_root = _write_refiner_index(tmp_path)
    pipe, runtime = _refined_pipeline(monkeypatch, tmp_path, events, pin_cpu_memory=pin_memory)

    pipe(_refined_request())
    pipe(_refined_request())

    assert len(runtime.calls) == 2
    (model,) = refiner_factory.models
    assert refiner_factory.requested_configs == [MAGI2_REFINER_CONFIG]
    assert model.checkpoints == [refiner_root]
    # The refiner is ready before the request generates anything.
    assert events == ["build", "load_checkpoint", ("stager", "_FakeRefiner"), "evaluate", "evaluate"]

    assert isinstance(pipe.refiner, _Magi2StagedComponent)
    assert pipe.refiner.module is model
    assert pipe._modules["refiner"] is pipe.refiner
    (stager,) = stagers
    assert stager.module is model
    assert stager.device == torch.device("cpu")
    assert stager.pin_memory is pin_memory

    assert pipe.refiner_sampler.model is model
    assert pipe.refiner_sampler.data_proxy is pipe.refiner_data_proxy
    assert pipe.refiner_data_proxy.config is model.config
    assert pipe.refiner_sampler.guidance_scale == 2.0


@pytest.mark.parametrize(("output_rank", "all_ranks"), [(True, False), (False, False), (False, True)])
def test_preview_is_released_and_reloaded_from_one_host_snapshot(monkeypatch, events, stagers, output_rank, all_ranks):
    pipe, _ = _pipeline()
    pipe.transformer = nn.Linear(2, 2)
    pipe.device_str = "cpu"
    pipe.od_config = _topology_config()
    pipe._is_output_rank = output_rank
    pipe._transformer_is_layerwise_offloaded = False
    monkeypatch.setattr(torch.accelerator, "empty_cache", lambda: None)

    for _ in range(2):
        pipe._release_preview(all_ranks=all_ranks)
        pipe._release_preview(all_ranks=all_ranks)
        events.append("released")
        pipe._ensure_preview_resident()
        pipe._ensure_preview_resident()
        events.append("resident")

    if not (output_rank or all_ranks):
        assert not stagers
        assert events == ["released", "resident", "released", "resident"]
        return
    (stager,) = stagers
    assert stager.module is pipe.transformer
    assert stager.pin_memory is True
    # One snapshot that keeps the device copy; a release drops it once and a reload copies it back once.
    assert stager.loaded is True
    assert events == [
        ("stager", "Linear"),
        ("stage_out", "Linear"),
        "released",
        ("stage_in", "Linear"),
        "resident",
        ("stage_out", "Linear"),
        "released",
        ("stage_in", "Linear"),
        "resident",
    ]


@pytest.mark.parametrize("musa", [False, True], ids=["cuda", "musa"])
def test_refiner_is_compiled_with_the_preview_region_options(
    monkeypatch, tmp_path, events, stagers, refiner_factory, musa
):
    _write_refiner_index(tmp_path)
    pipe, _ = _refined_pipeline(monkeypatch, tmp_path, events)
    options = {"fullgraph": True, "dynamic": False, "options": {"emulate_precision_casts": True}}
    pipe._region_compile_kwargs = dict(options)
    monkeypatch.setattr(pipeline.current_omni_platform, "is_musa", lambda: musa)

    pipe(_refined_request())

    expected = dict(options)
    if musa:
        # MUSA keeps the refiner's MLP activations unpadded; the Preview options are untouched.
        padding = {"comprehensive_padding": False, "shape_padding": False}
        expected["options"] = {**options["options"], **padding}
    assert events[:4] == ["build", "load_checkpoint", ("compile_regions", expected), ("stager", "_FakeRefiner")]
    assert pipe._region_compile_kwargs == options


def test_refiner_is_built_outside_inference_mode_under_the_pipeline_config(
    monkeypatch, tmp_path, events, stagers, refiner_factory
):
    _write_refiner_index(tmp_path)
    pipe, _ = _refined_pipeline(monkeypatch, tmp_path, events)

    pipe(_refined_request())

    (model,) = refiner_factory.models
    assert model.built_in_inference_mode is False
    assert model.loaded_in_inference_mode is False
    assert not model.weight.is_inference()
    assert model.built_under_config is pipe.od_config
    assert get_current_diffusion_config_or_none() is None


@pytest.mark.parametrize("index", [False, True], ids=["no-refiner-dir", "no-index"])
def test_refined_request_without_the_refiner_checkpoint_fails_before_generation(
    monkeypatch, tmp_path, events, stagers, refiner_factory, index
):
    if index:
        (tmp_path / "refiner").mkdir()
    pipe, runtime = _refined_pipeline(monkeypatch, tmp_path, events)

    with pytest.raises(OmniClientError, match=r"needs the refiner checkpoint.*refiner/model\.safetensors\.index\.json"):
        pipe(_refined_request())

    assert not runtime.calls
    assert not refiner_factory.models
    assert not stagers
    assert getattr(pipe, "refiner", None) is None
    # A Preview request still runs, and the refiner is picked up once it exists.
    pipe(_request("A fox walks through snow"))
    _write_refiner_index(tmp_path)
    pipe(_refined_request())
    assert len(refiner_factory.models) == 1
    assert len(runtime.calls) == 2


@pytest.mark.parametrize(
    ("topology", "message"),
    [
        ({"tensor_parallel_size": 2}, "needs tensor_parallel_size=1; got 2"),
        ({"sequence_parallel_size": 3, "ulysses_degree": 3}, r"sequence_parallel_size=3 does not divide"),
        ({"sequence_parallel_size": 16, "ulysses_degree": 16}, r"sequence_parallel_size=16 does not divide"),
        ({"enable_layerwise_offload": True}, "with layer_wise offload of the Preview"),
        (
            {"enable_distributed_layerwise_offload": True, "dlo_use_allgather": False},
            "with distributed_layer_wise offload of the Preview",
        ),
        ({"use_hsdp": True, "sequence_parallel_size": 8, "ulysses_degree": 8}, "with HSDP of the Preview"),
    ],
)
def test_refined_request_rejects_topologies_the_refiner_cannot_serve(
    monkeypatch, tmp_path, events, stagers, refiner_factory, topology, message
):
    _write_refiner_index(tmp_path)
    pipe, runtime = _refined_pipeline(monkeypatch, tmp_path, events, **topology)

    with pytest.raises(OmniClientError, match=message) as error:
        pipe(_refined_request())

    assert "request 272p or 540p" in str(error.value)
    assert not runtime.calls
    assert not refiner_factory.models
    assert not stagers
    # The Preview resolutions keep working on the same deployment.
    pipe(_request("A fox walks through snow", extra_args={"resolution": "540p"}))
    assert len(runtime.calls) == 1


@pytest.mark.parametrize(
    "topology",
    [
        {},
        {"sequence_parallel_size": 2, "ulysses_degree": 2},
        {"sequence_parallel_size": 4, "ulysses_degree": 4, "cfg_parallel_size": 2},
        {"sequence_parallel_size": 8, "ulysses_degree": 8},
        {"sequence_parallel_size": 8, "ulysses_degree": 8, "vae_patch_parallel_size": 8},
    ],
)
def test_refined_request_accepts_replicated_sequence_parallel_topologies(
    monkeypatch, tmp_path, events, stagers, refiner_factory, topology
):
    _write_refiner_index(tmp_path)
    pipe, runtime = _refined_pipeline(monkeypatch, tmp_path, events, **topology)

    pipe(_refined_request())

    assert len(refiner_factory.models) == 1
    assert len(runtime.calls) == 1


class _FakeComponent(nn.Module):
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))


def _initialized_pipeline(monkeypatch: pytest.MonkeyPatch, root: Path, **config: Any) -> Magi2Pipeline:
    """Run Magi2Pipeline.__init__ with fake components on an SP8 configuration."""

    monkeypatch.setenv("MAGI2_DETERMINISTIC", "0")
    monkeypatch.setattr(
        pipeline,
        "current_omni_platform",
        SimpleNamespace(device_type="cuda", is_available=lambda: True, is_cuda=lambda: True, is_musa=lambda: False),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: 0)
    monkeypatch.setattr(pipeline, "_resolve_checkpoint_root", lambda model, revision: str(root))
    monkeypatch.setattr(modeling_magi2, "Magi2PreviewTransformer", _FakeComponent)
    monkeypatch.setattr(pipeline, "Magi2Qwen35TextEncoder", _FakeComponent)
    monkeypatch.setattr(pipeline, "Magi2AudioDecoder", _FakeComponent)
    monkeypatch.setattr(pipeline, "Magi2TurboVAEDecoder", _FakeComponent)
    monkeypatch.setattr("vllm_omni.diffusion.models.lance.wan_vae.LanceWanVAE", _FakeComponent)
    od_config = OmniDiffusionConfig(
        model=str(root),
        parallel_config=DiffusionParallelConfig(ulysses_degree=8),
        **config,
    )
    return Magi2Pipeline(od_config=od_config)


def test_pipeline_initialization_builds_nothing_of_the_refiner(monkeypatch, tmp_path, events, stagers, refiner_factory):
    _write_refiner_index(tmp_path)

    pipe = _initialized_pipeline(monkeypatch, tmp_path)

    assert not refiner_factory.models
    assert pipe.refiner is None
    assert pipe.refiner_data_proxy is None
    assert pipe.refiner_sampler is None
    assert "refiner" not in pipe._modules
    # Text encoder, image VAE, video decoder and audio decoder only.
    assert len(stagers) == 4
    assert all(not isinstance(stager.module, _FakeRefiner) for stager in stagers)


# The refined geometry is scaled down so the pipeline runs on the CPU: a
# one-second clip gives a three-frame Preview latent at 64x32 and a
# five-frame refiner latent at 128x64.
_PREVIEW_SIZE = (64, 32)
_REFINED_SIZE = (128, 64)
_PREVIEW_LATENT = (1, 48, 3, 2, 4)
_AUDIO_LATENT = (1, 25, 64)
_REFINER_GRID = (5, 4, 8)
_COND_LEN, _UNCOND_LEN = 7, 3


class _FakePreviewSampler:
    """Returns the initial noise unchanged, so the refiner sees the Preview's own draws."""

    def __init__(self, events: list[Any]) -> None:
        self.events = events
        self.outputs: list[tuple[torch.Tensor, torch.Tensor]] = []

    def sample(self, sampler_input) -> tuple[torch.Tensor, torch.Tensor]:
        self.events.append("preview")
        output = (sampler_input.latent, sampler_input.audio_latent)
        self.outputs.append(output)
        return output


@dataclasses.dataclass
class _Runtime:
    events: list[Any]
    sampler: _FakePreviewSampler
    context: torch.Tensor
    context_null: torch.Tensor
    decoded_video: list[torch.Tensor] = dataclasses.field(default_factory=list)
    decoded_audio: list[torch.Tensor] = dataclasses.field(default_factory=list)
    encoded_images: list[tuple[int, int]] = dataclasses.field(default_factory=list)
    prewarm_calls: list[tuple[Any, ...]] = dataclasses.field(default_factory=list)
    renoise_calls: list[tuple[torch.Tensor, tuple[int, ...]]] = dataclasses.field(default_factory=list)


@pytest.fixture
def small_geometry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pipeline, "_RESOLUTION_PRESETS", {"540p": _PREVIEW_SIZE, "1080p": _REFINED_SIZE})
    monkeypatch.setattr(pipeline, "_PRESET_BY_SIZE", {_PREVIEW_SIZE: "540p", _REFINED_SIZE: "1080p"})
    monkeypatch.setattr(pipeline, "_REFINER_PREVIEW_SIZES", {"1080p": _PREVIEW_SIZE})
    monkeypatch.setattr(
        pipeline,
        "MAGI2_GENERATION_CONFIG",
        dataclasses.replace(pipeline.MAGI2_GENERATION_CONFIG, duration_seconds=1.0, output_frames=12),
    )


@pytest.fixture
def runtime(monkeypatch: pytest.MonkeyPatch, events: list[Any], small_geometry: None) -> _Runtime:
    generator = torch.Generator().manual_seed(1)
    context = (torch.randn(1, _COND_LEN, _TEXT_CHANNELS, generator=generator) + 1.0).to(torch.bfloat16)
    context_null = (torch.randn(1, _UNCOND_LEN, _TEXT_CHANNELS, generator=generator) - 1.0).to(torch.bfloat16)
    state = _Runtime(events=events, sampler=_FakePreviewSampler(events), context=context, context_null=context_null)

    def prewarm(*args, **kwargs):
        events.append("prewarm")
        state.prewarm_calls.append((*args, kwargs.get("window")))

    real_renoise = pipeline.upsample_and_renoise

    def renoise(latent, size, *args, **kwargs):
        events.append("renoise")
        state.renoise_calls.append((latent, tuple(size)))
        return real_renoise(latent, size, *args, **kwargs)

    monkeypatch.setattr(pipeline, "prewarm_window_attention", prewarm)
    monkeypatch.setattr(pipeline, "_PREWARMED_WINDOW_ATTENTION", set())
    monkeypatch.setattr(pipeline, "upsample_and_renoise", renoise)
    monkeypatch.setattr(pipeline.current_omni_platform, "is_available", lambda: False)
    return state


def _runtime_pipeline(root: Path, state: _Runtime) -> Magi2Pipeline:
    """A pipeline whose encoders, Preview sampler and decoders are fakes; the refiner stage is real."""

    pipe = Magi2Pipeline.__new__(Magi2Pipeline)
    nn.Module.__init__(pipe)
    pipe.od_config = _topology_config()
    pipe.checkpoint_root = str(root)
    pipe.dtype = torch.bfloat16
    pipe.deterministic = False
    pipe.enable_diffusion_pipeline_profiler = False
    pipe._is_output_rank = True
    pipe._offload_aux_after_use = True
    pipe.refiner = None
    pipe.refiner_data_proxy = None
    pipe.refiner_sampler = None
    return _install_runtime_fakes(pipe, state)


def _install_runtime_fakes(pipe: Magi2Pipeline, state: _Runtime) -> Magi2Pipeline:
    """Run ``pipe`` on the CPU with fake encoders, Preview sampler and decoders."""

    events = state.events
    pipe.device_str = "cpu"
    pipe.sampler = state.sampler
    # Preview residency changes, with the number of runtime events seen so far.
    pipe.residency = []
    pipe._release_preview = MethodType(
        lambda _self, *, all_ranks=False: _self.residency.append(("release", all_ranks, len(events))), pipe
    )
    pipe._ensure_preview_resident = MethodType(lambda _self: _self.residency.append(("resident", len(events))), pipe)

    def encode_prompts(_self, prompts):
        events.append("encode")
        assert len(prompts) == 2
        return state.context, state.context_null

    def encode_reference_image(_self, image, height, width):
        state.encoded_images.append((height, width))
        return torch.zeros(1, 1, 48, 1, height // 16, width // 16), torch.tensor([[[height // 16, width // 16]]])

    def decode_video(_self, latent):
        events.append("decode_video")
        state.decoded_video.append(latent)
        return np.zeros((2, 4, 6, 3), dtype=np.uint8)

    def decode_audio(_self, latent):
        events.append("decode_audio")
        state.decoded_audio.append(latent)
        return np.zeros((8, 2), dtype=np.float32)

    pipe._encode_prompts = MethodType(encode_prompts, pipe)
    pipe._encode_reference_image = MethodType(encode_reference_image, pipe)
    pipe._pool_figure_token = MethodType(lambda _self, prompt, context: torch.zeros(1, _TEXT_CHANNELS), pipe)
    pipe._decode_video = MethodType(decode_video, pipe)
    pipe._decode_audio = MethodType(decode_audio, pipe)
    return pipe


def _record_plans(pipe: Magi2Pipeline, events: list[Any]) -> list[Any]:
    proxy = pipe.refiner_data_proxy
    assert proxy is not None
    built = proxy.window_plan
    plans: list[Any] = []

    def window_plan(grid, text_len, device="cpu"):
        events.append(("plan", int(text_len)))
        plan = built(grid, text_len, device)
        plans.append(plan)
        return plan

    proxy.window_plan = window_plan
    return plans


def _record_refiner_samples(pipe: Magi2Pipeline, events: list[Any]) -> list[torch.Tensor]:
    sampler = pipe.refiner_sampler
    assert sampler is not None
    sample = sampler.sample
    outputs: list[torch.Tensor] = []

    def recording_sample(*args, **kwargs):
        events.append(("refiner_sample", kwargs["num_inference_steps"]))
        output = sample(*args, **kwargs)
        outputs.append(output)
        return output

    sampler.sample = recording_sample
    return outputs


def _refiner_step_events(steps: int) -> list[Any]:
    one_step = [("plan", _COND_LEN), ("refiner", _COND_LEN), ("plan", _UNCOND_LEN), ("refiner", _UNCOND_LEN)]
    return one_step * steps


def test_refined_request_runs_preview_renoise_refiner_steps_then_decode(
    tmp_path, events, stagers, refiner_factory, runtime
):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)
    pipe._load_refiner()
    plans = _record_plans(pipe, events)
    refined = _record_refiner_samples(pipe, events)
    del events[:]

    result = pipe(_request("A fox walks through snow", extra_args={"resolution": "1080p"}))

    # The Preview leaves for the prompt encoding, returns for its denoise and
    # leaves every rank before the refiner and the decode.
    assert pipe.residency == [("release", False, 0), ("resident", 1), ("release", True, 2)]
    assert events == [
        "encode",
        "preview",
        ("stage_in", "_FakeRefiner"),
        "prewarm",
        ("plan", _COND_LEN),
        ("plan", _UNCOND_LEN),
        "renoise",
        ("refiner_sample", 5),
        *_refiner_step_events(5),
        ("stage_out", "_FakeRefiner"),
        "decode_video",
        "decode_audio",
    ]
    # Every refiner forward reuses the two plans built before the loop.
    (model,) = refiner_factory.models
    assert all(call["window_plan"] is plans[0 if i % 2 == 0 else 1] for i, call in enumerate(model.calls))
    assert [plan.grid for plan in plans[:2]] == [_REFINER_GRID, _REFINER_GRID]

    (preview_latent, preview_audio) = runtime.sampler.outputs[0]
    assert tuple(preview_latent.shape) == _PREVIEW_LATENT
    ((renoise_input, renoise_size),) = runtime.renoise_calls
    assert renoise_input is preview_latent and renoise_size == _REFINER_GRID
    (decoded,) = runtime.decoded_video
    assert decoded is refined[0]
    assert decoded.dtype == torch.float32 and tuple(decoded.shape) == (1, 48, *_REFINER_GRID)
    # The Preview audio latent is decoded as it left the Preview sampler.
    assert runtime.decoded_audio == [preview_audio]
    assert runtime.decoded_audio[0] is preview_audio

    assert result.output["metadata"]["video"]["fps"] == 25.0


def test_refiner_noise_is_the_third_draw_of_the_request_generator(tmp_path, events, stagers, refiner_factory, runtime):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)

    pipe(_request("A fox walks through snow", seed=123, extra_args={"resolution": "1080p"}))

    torch.manual_seed(123)
    video = torch.randn(_PREVIEW_LATENT)
    audio = torch.randn(_AUDIO_LATENT)
    upsampled = F.interpolate(video, size=_REFINER_GRID, mode="trilinear", align_corners=True)
    noise = torch.randn(upsampled.shape)
    signal, noise_scale = zero_snr_signal_coefficients()
    expected = upsampled * signal + noise * noise_scale

    (preview_latent, preview_audio) = runtime.sampler.outputs[0]
    assert torch.equal(preview_latent, video) and torch.equal(preview_audio, audio)
    first_call = refiner_factory.models[0].calls[0]
    assert pipe.refiner_data_proxy is not None
    assert torch.equal(first_call["video"], pipe.refiner_data_proxy.pack(expected, runtime.context).video)


def test_refined_latent_follows_the_guided_refiner_schedule(tmp_path, events, stagers, refiner_factory, runtime):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)

    extra_args = {"resolution": "1080p", "refiner_num_inference_steps": 3}
    pipe(_request("A fox walks through snow", seed=123, extra_args=extra_args))
    (refined,) = runtime.decoded_video

    # Replay the renoised latent, then step the refiner's scheduler (shift 5)
    # by hand with the guided velocity u + 2 * (c - u).
    torch.manual_seed(123)
    video = torch.randn(_PREVIEW_LATENT)
    torch.randn(_AUDIO_LATENT)
    upsampled = F.interpolate(video, size=_REFINER_GRID, mode="trilinear", align_corners=True)
    signal, noise_scale = zero_snr_signal_coefficients()
    latent = upsampled * signal + torch.randn(upsampled.shape) * noise_scale
    sampler = pipe.refiner_sampler
    assert sampler is not None
    scheduler = build_magi2_refiner_scheduler(3, shift=5.0, device="cpu")
    for timestep in scheduler.timesteps:
        cond = sampler.predict_noise(latent, runtime.context)
        uncond = sampler.predict_noise(latent, runtime.context_null)
        velocity = (uncond + 2.0 * (cond - uncond)).to(torch.float32)
        latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]

    assert torch.equal(refined, latent)


def test_refiner_is_staged_once_per_request_and_prewarmed_once_per_process(
    tmp_path, events, stagers, refiner_factory, runtime
):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)

    for _ in range(2):
        pipe(_request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 3}))

    assert events.count(("stage_in", "_FakeRefiner")) == 2
    assert events.count(("stage_out", "_FakeRefiner")) == 2
    first, second = (events[i:] for i in [i for i, event in enumerate(events) if event == "encode"])
    for request in (first, second):
        stage_in = request.index(("stage_in", "_FakeRefiner"))
        stage_out = request.index(("stage_out", "_FakeRefiner"))
        refiner_calls = [i for i, event in enumerate(request[:stage_out]) if event[0] == "refiner"]
        assert len(refiner_calls) == 6
        assert stage_in < refiner_calls[0] and refiner_calls[-1] < stage_out < request.index("decode_video")
    assert runtime.prewarm_calls == [(torch.device("cpu"), 4, 2, 32, torch.bfloat16, _TINY_CONFIG.window)]

    # A second pipeline in the same process finds the kernels compiled.
    other = _runtime_pipeline(tmp_path, runtime)
    other(_request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 1}))
    assert len(runtime.prewarm_calls) == 1
    assert len(refiner_factory.models) == 2


def test_refiner_prewarm_uses_this_ranks_heads(monkeypatch, tmp_path, events, stagers, refiner_factory, runtime):
    _write_refiner_index(tmp_path)
    monkeypatch.setattr(pipeline, "get_magi2_ulysses_group", lambda: Magi2ParallelGroup(None, 2, 0))
    pipe = _runtime_pipeline(tmp_path, runtime)

    pipe(_request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 1}))

    assert runtime.prewarm_calls == [(torch.device("cpu"), 2, 1, 32, torch.bfloat16, _TINY_CONFIG.window)]


def test_preview_request_after_a_refined_one_leaves_the_refiner_alone(
    tmp_path, events, stagers, refiner_factory, runtime
):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)
    pipe(_request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 1}))
    calls = len(refiner_factory.models[0].calls)
    del events[:]

    result = pipe(_request("A fox walks through snow", extra_args={"resolution": "540p"}))

    assert events == ["encode", "preview", "decode_video", "decode_audio"]
    assert len(refiner_factory.models[0].calls) == calls
    assert len(runtime.renoise_calls) == 1
    (preview_latent, preview_audio) = runtime.sampler.outputs[-1]
    assert runtime.decoded_video[-1] is preview_latent
    assert runtime.decoded_audio[-1] is preview_audio
    assert result.output["metadata"]["video"]["fps"] == 12.5


def test_refined_image_request_encodes_the_image_at_the_preview_size(
    tmp_path, events, stagers, refiner_factory, runtime
):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)
    image = Image.new("RGB", (40, 30), "white")

    pipe(
        _request(
            {"prompt": "The first frame begins moving", "multi_modal_data": {"image": image}},
            extra_args={"resolution": "1080p", "refiner_num_inference_steps": 1},
        )
    )

    width, height = _PREVIEW_SIZE
    assert runtime.encoded_images == [(height, width)]
    (decoded,) = runtime.decoded_video
    assert tuple(decoded.shape) == (1, 48, *_REFINER_GRID)


def test_refiner_stages_are_profiled(tmp_path, events, stagers, refiner_factory, runtime):
    _write_refiner_index(tmp_path)
    pipe = _runtime_pipeline(tmp_path, runtime)
    assert "_evaluate_refiner" in pipe._PROFILER_TARGETS
    pipe.setup_diffusion_pipeline_profiler(
        profiler_targets=list(pipe._PROFILER_TARGETS),
        enable_diffusion_pipeline_profiler=True,
    )

    refined = pipe(
        _request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 2})
    )
    preview = pipe(_request("A fox walks through snow", extra_args={"resolution": "540p"}))

    refiner_stages = {
        "Magi2Pipeline._evaluate_refiner",
        "Magi2Pipeline.refiner_sampler.sample",
        "Magi2Pipeline.refiner_sampler.diffuse",
    }
    assert refiner_stages <= set(refined.stage_durations)
    assert {"Magi2Pipeline.sampler.sample", "Magi2Pipeline._decode_video"} <= set(refined.stage_durations)
    assert not refiner_stages & set(preview.stage_durations)
    sampler = pipe.refiner_sampler
    assert sampler is not None
    # Wrapped once: the wrapper calls the sampler's own method directly.
    assert sampler.sample.__wrapped__.__func__ is pipeline.Magi2RefinerSampler.sample


def test_initialization_profiles_the_refiner_stage(monkeypatch, tmp_path, events, stagers, refiner_factory, runtime):
    _write_refiner_index(tmp_path)
    pipe = _initialized_pipeline(monkeypatch, tmp_path, enable_diffusion_pipeline_profiler=True)

    for path in pipe._PROFILER_TARGETS:
        assert hasattr(functools.reduce(getattr, path.split("."), pipe), "__wrapped__"), path
    monkeypatch.setattr(
        pipeline,
        "current_omni_platform",
        SimpleNamespace(device_type="cpu", is_available=lambda: False, is_cuda=lambda: False, is_musa=lambda: False),
    )
    _install_runtime_fakes(pipe, runtime)

    refined = pipe(
        _request("A fox walks through snow", extra_args={"resolution": "1080p", "refiner_num_inference_steps": 1})
    )

    assert "Magi2Pipeline._evaluate_refiner" in refined.stage_durations
    assert len(refiner_factory.models[0].calls) == 2
