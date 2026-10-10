# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import dataclasses
from typing import Any

import numpy as np
import PIL.Image
import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.video_processor import VideoProcessor
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.request import DUMMY_DIFFUSION_REQUEST_ID, OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.errors import OmniClientError
from vllm_omni.inputs.data import OmniDiffusionSamplingParams, OmniPromptType

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StopAtLatentsError(Exception):
    pass


class _FakeTransformer:
    """The attributes `PAN2Pipeline.forward` reads from the transformer before latent preparation.

    Not an `nn.Module`: the pipeline under test is built without `nn.Module.__init__`, so it cannot hold submodules.
    """

    patch_size = (1, 2, 2)

    def __init__(self):
        self.x_embedder = nn.Linear(1, 1)


def _make_pipeline():
    """A `PAN2Pipeline` without weights whose denoising steps leave the latents unchanged; the VAE is never set."""
    from vllm_omni.diffusion.cache.cachedit import RequestScopedCacheDiTRuntime
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline
    from vllm_omni.diffusion.models.pan2.quality_policy import PAN2QualityPolicy

    pipeline = object.__new__(PAN2Pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vae_scale_factor_spatial = 16
    pipeline.vae_scale_factor_temporal = 4
    pipeline.num_channels_latents = 4
    pipeline.sequential_cfg_branches = False
    pipeline._quality_policy = PAN2QualityPolicy(OmniDiffusionConfig(cache_backend="none"))
    pipeline._cache_dit_runtime = RequestScopedCacheDiTRuntime(pipeline)
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler(shift=7.0)
    pipeline.encode_prompt = lambda _prompt, _device: torch.zeros(1, 1, 4)
    pipeline.predict_noise_maybe_with_cfg = lambda **kwargs: torch.zeros_like(
        kwargs["positive_kwargs"]["hidden_states"]
    )
    pipeline.scheduler_step_maybe_with_cfg = lambda noise_pred, t, latents, do_true_cfg: latents
    return pipeline


def _resolved_num_frames(request_id="pan2-test", **sampling_overrides) -> int:
    """Run `PAN2Pipeline.forward` up to latent preparation and return the number of frames it resolved."""
    pipeline = _make_pipeline()

    def prepare_latents(batch_size, height, width, num_frames, device, generator, latents):
        pipeline.num_frames = num_frames
        raise _StopAtLatentsError

    pipeline.prepare_latents = prepare_latents

    sampling_params = OmniDiffusionSamplingParams(height=64, width=64, num_inference_steps=2, **sampling_overrides)
    request = OmniDiffusionRequest(prompt="a cat", sampling_params=sampling_params, request_id=request_id)
    with pytest.raises(_StopAtLatentsError):
        pipeline.forward(DiffusionRequestBatch([request]))
    return pipeline.num_frames


def test_omitted_num_frames_uses_the_pan2_default():
    from vllm_omni.diffusion.models.pan2.pipeline_pan2 import PAN2_DEFAULT_NUM_FRAMES

    assert OmniDiffusionSamplingParams().num_frames == 1
    assert _resolved_num_frames() == PAN2_DEFAULT_NUM_FRAMES


def test_requested_num_frames_is_kept():
    assert _resolved_num_frames(num_frames=9) == 9


def test_dummy_run_keeps_a_single_frame():
    assert _resolved_num_frames(request_id=DUMMY_DIFFUSION_REQUEST_ID, num_frames=1) == 1


def _generate_latents(pipeline, prompt: OmniPromptType = "a cat", **sampling_overrides) -> torch.Tensor:
    sampling_params = OmniDiffusionSamplingParams(
        height=64, width=64, num_frames=5, num_inference_steps=1, output_type="latent", **sampling_overrides
    )
    request = OmniDiffusionRequest(prompt=prompt, sampling_params=sampling_params, request_id="pan2-test")
    return pipeline.forward(DiffusionRequestBatch([request])).output


def test_request_output_type_latent_returns_the_latents():
    assert _generate_latents(_make_pipeline()).shape == (1, 4, 2, 4, 4)


def test_num_outputs_per_prompt_batches_the_videos_with_one_generator_each():
    pipeline = _make_pipeline()
    calls: list[dict[str, Any]] = []

    def predict_noise_maybe_with_cfg(**kwargs: Any) -> torch.Tensor:
        calls.append(kwargs)
        return torch.zeros_like(kwargs["positive_kwargs"]["hidden_states"])

    pipeline.predict_noise_maybe_with_cfg = predict_noise_maybe_with_cfg

    latents = _generate_latents(
        pipeline, num_outputs_per_prompt=2, guidance_scale=3.0, generator=torch.Generator().manual_seed(0)
    )

    assert latents.shape == (2, 4, 2, 4, 4)
    (call,) = calls
    assert call["positive_kwargs"]["hidden_states"].shape[0] == 2
    assert call["positive_kwargs"]["encoder_hidden_states"].shape[0] == 2
    assert call["negative_kwargs"]["encoder_hidden_states"].shape[0] == 2
    # The request generator draws the noise of each video in turn.
    generator = torch.Generator().manual_seed(0)
    expected = [torch.randn(1, 4, 2, 4, 4, generator=generator) for _ in range(2)]
    torch.testing.assert_close(latents, torch.cat(expected), rtol=0, atol=0)


@pytest.mark.parametrize("num_outputs", [-1, 11])
def test_out_of_range_num_outputs_per_prompt_is_rejected(num_outputs: int):
    with pytest.raises(OmniClientError, match="num_outputs_per_prompt must be in"):
        _generate_latents(_make_pipeline(), num_outputs_per_prompt=num_outputs)


@pytest.mark.parametrize(
    "sampling_overrides", [{"sigmas": [1.0, 0.5]}, {"timesteps": torch.tensor([999.0, 500.0])}], ids=str
)
def test_custom_sigmas_and_timesteps_are_rejected(sampling_overrides: dict[str, Any]):
    with pytest.raises(OmniClientError, match="custom `sigmas`/`timesteps` are not supported"):
        _generate_latents(_make_pipeline(), **sampling_overrides)


def test_video_input_is_rejected():
    prompt: OmniPromptType = {"prompt": "a cat", "multi_modal_data": {"video": [PIL.Image.new("RGB", (64, 64))]}}
    with pytest.raises(OmniClientError, match="PAN2 does not accept video input"):
        _generate_latents(_make_pipeline(), prompt=prompt)


@pytest.mark.parametrize(
    ("prompt", "sampling_overrides", "match"),
    [
        ("a cat", {"sigmas": [1.0, 0.5]}, "custom `sigmas`/`timesteps` are not supported"),
        (
            {"prompt": "a cat", "multi_modal_data": {"video": [PIL.Image.new("RGB", (64, 64))]}},
            {},
            "PAN2 does not accept video input",
        ),
    ],
    ids=["sigmas", "video"],
)
def test_unsupported_requests_are_rejected_before_dispatch(
    prompt: OmniPromptType, sampling_overrides: dict[str, Any], match: str
):
    # The engine-process pre-process rejects them, so the client error keeps its 4xx status under multi-GPU executors.
    from vllm_omni.diffusion.models.pan2 import get_pan2_pre_process_func

    pre_process = get_pan2_pre_process_func(OmniDiffusionConfig(model_config={"guardrails": False}))
    request = OmniDiffusionRequest(
        prompt=prompt, sampling_params=OmniDiffusionSamplingParams(**sampling_overrides), request_id="pan2-test"
    )
    with pytest.raises(OmniClientError, match=match):
        pre_process(request)


@dataclasses.dataclass
class _FakeLatentDist:
    latents: torch.Tensor

    def mode(self) -> torch.Tensor:
        return self.latents


@dataclasses.dataclass
class _FakeEncoderOutput:
    latent_dist: _FakeLatentDist


@dataclasses.dataclass
class _FakeImageVAEConfig:
    latents_mean: tuple[float, ...] = (0.0,) * 4
    latents_std: tuple[float, ...] = (1.0,) * 4


class _FakeImageVAE:
    """Encodes every image to latents of ones and counts its calls."""

    dtype = torch.float32
    config = _FakeImageVAEConfig()

    def __init__(self):
        self.num_encodes = 0

    def encode(self, image: torch.Tensor) -> _FakeEncoderOutput:
        self.num_encodes += 1
        batch_size, _, num_frames, height, width = image.shape
        return _FakeEncoderOutput(_FakeLatentDist(torch.ones(batch_size, 4, num_frames, height // 16, width // 16)))


def test_image_condition_is_encoded_once_for_every_video():
    pipeline = _make_pipeline()
    pipeline.vae = _FakeImageVAE()
    pipeline.video_processor = VideoProcessor(vae_scale_factor=16)

    latents = torch.zeros(2, 4, 2, 4, 4)
    condition_latents = pipeline.prepare_condition_latents(latents, PIL.Image.new("RGB", (96, 64)), 64, 64)

    assert pipeline.vae.num_encodes == 1
    assert condition_latents.shape == (2, 5, 2, 4, 4)
    assert (condition_latents[:, :, :1] == 1).all()
    assert (condition_latents[:, :, 1:] == 0).all()


@pytest.mark.parametrize("num_videos", [1, 2])
def test_post_process_returns_every_video(num_videos: int):
    from vllm_omni.diffusion.models.pan2.pipeline_pan2 import get_pan2_post_process_func

    post_process = get_pan2_post_process_func(OmniDiffusionConfig(model_config={"guardrails": False}))
    video = torch.zeros(num_videos, 3, 5, 16, 16)

    frames = post_process(video, output_type="np")
    assert isinstance(frames, np.ndarray)
    assert frames.shape == (num_videos, 5, 16, 16, 3)

    videos = post_process(video, output_type="pil")
    if num_videos == 1:
        # A single video keeps the frame list existing clients read.
        videos = [videos]
    assert len(videos) == num_videos
    for frames in videos:
        assert isinstance(frames, list) and len(frames) == 5
        assert all(isinstance(frame, PIL.Image.Image) for frame in frames)


class _FakeComponent:
    """A loaded tokenizer, text encoder or scheduler: only `.to` is used at construction."""

    def to(self, *args, **kwargs):
        return self


@dataclasses.dataclass
class _FakeVAEConfig:
    scale_factor_spatial: int = 16
    scale_factor_temporal: int = 4
    z_dim: int = 48


class _FakeVAE(_FakeComponent):
    config = _FakeVAEConfig()


class _FakePAN2Transformer:
    def __init__(self, **kwargs):
        from vllm_omni.diffusion.models.pan2 import PAN2Transformer3DModel

        self._cache_dit_adapter_config = PAN2Transformer3DModel._cache_dit_adapter_config


def test_every_component_loads_the_requested_revision(monkeypatch: pytest.MonkeyPatch):
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline, pipeline_pan2

    calls: list[tuple[str, str | None]] = []

    class _Loader:
        def __init__(self, name: str, component: _FakeComponent):
            self.name = name
            self.component = component

        def from_pretrained(self, *args, revision: str | None = None, **kwargs) -> _FakeComponent:
            calls.append((self.name, revision))
            return self.component

    def prefetch_subfolders(model, subfolders, *, revision=None, **kwargs) -> None:
        calls.append(("prefetch", revision))

    def from_pretrained_with_prefetch(factory, model, *, subfolder, revision=None, **kwargs) -> _FakeComponent:
        calls.append((subfolder, revision))
        return _FakeVAE() if subfolder == "vae" else _FakeComponent()

    monkeypatch.setattr(pipeline_pan2, "prefetch_subfolders", prefetch_subfolders)
    monkeypatch.setattr(pipeline_pan2, "from_pretrained_with_prefetch", from_pretrained_with_prefetch)
    monkeypatch.setattr(pipeline_pan2, "AutoTokenizer", _Loader("tokenizer", _FakeComponent()))
    monkeypatch.setattr(pipeline_pan2, "FlowMatchEulerDiscreteScheduler", _Loader("scheduler", _FakeComponent()))
    monkeypatch.setattr(pipeline_pan2, "PAN2Transformer3DModel", _FakePAN2Transformer)

    pipeline = PAN2Pipeline(od_config=OmniDiffusionConfig(model="IFM/PAN2", revision="abc123"))

    assert sorted(calls) == sorted(
        (name, "abc123") for name in ("prefetch", "tokenizer", "text_encoder", "vae", "scheduler")
    )
    assert [source.revision for source in pipeline.weights_sources] == ["abc123"]


@pytest.mark.parametrize("degree", ["ring_degree", "allgather_degree"])
def test_ring_and_allgather_sequence_parallel_are_rejected_before_loading(degree: str):
    from vllm_omni.diffusion.data import DiffusionParallelConfig
    from vllm_omni.diffusion.models.pan2 import PAN2Pipeline

    od_config = OmniDiffusionConfig(model="unused", parallel_config=DiffusionParallelConfig(**{degree: 2}))
    with pytest.raises(NotImplementedError, match="ring or all-gather sequence parallel"):
        PAN2Pipeline(od_config=od_config)
