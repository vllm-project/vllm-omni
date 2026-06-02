# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import sys
import types
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from PIL import Image
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class StubScheduler:
    def __init__(self, timesteps: list[int] | None = None, *, flow_shift: float = 1.0) -> None:
        self.timesteps = torch.tensor(timesteps or [9, 3], dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000, flow_shift=flow_shift)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []
        self.step_calls: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))
        self.timesteps = torch.arange(num_steps, 0, -1, dtype=torch.int64, device=device)

    def step(self, noise_pred: torch.Tensor, timestep: torch.Tensor, latents: torch.Tensor, **kwargs):
        del kwargs
        self.step_calls.append((noise_pred.clone(), timestep.clone(), latents.clone()))
        return (latents + noise_pred,)


class _ModeLatentDist:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class StubCosmos3VAE:
    dtype = torch.float32

    def __init__(self, z_dim: int = 2, *, temporal: int = 4, spatial: int = 8) -> None:
        self.config = SimpleNamespace(
            z_dim=z_dim,
            scale_factor_temporal=temporal,
            scale_factor_spatial=spatial,
            latents_mean=[0.0] * z_dim,
            latents_std=[1.0] * z_dim,
        )

    def encode(self, video: torch.Tensor):
        latent_frames = (video.shape[2] - 1) // self.config.scale_factor_temporal + 1
        latent_height = video.shape[-2] // self.config.scale_factor_spatial
        latent_width = video.shape[-1] // self.config.scale_factor_spatial
        latents = torch.ones(
            video.shape[0],
            self.config.z_dim,
            latent_frames,
            latent_height,
            latent_width,
            dtype=video.dtype,
            device=video.device,
        )
        return SimpleNamespace(latent_dist=_ModeLatentDist(latents))

    def decode(self, latents: torch.Tensor, return_dict: bool = False):
        del return_dict
        return (latents,)


class StubCosmos3Transformer(nn.Module):
    def __init__(
        self,
        *,
        latent_channel_size: int = 2,
    ) -> None:
        super().__init__()
        self.latent_channel_size = latent_channel_size
        self.cached_kv: Any | None = None
        self.cached_freqs_gen: Any | None = None
        self.calls: list[dict[str, Any]] = []
        self.reset_calls = 0

    def reset_cache(self) -> None:
        self.reset_calls += 1
        self.cached_kv = None
        self.cached_freqs_gen = None

    def forward(
        self,
        *,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        text_ids: torch.Tensor,
        text_mask: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        token = int(text_ids.reshape(-1)[0].item()) if text_ids.numel() else 0
        self.calls.append(
            {
                "token": token,
                "timestep": timestep.clone(),
                "text_mask": text_mask.clone(),
                "cache_before": self.cached_kv,
                "kwargs": dict(kwargs),
            }
        )
        if self.cached_kv is None:
            marker = torch.tensor([token], dtype=torch.float32)
            self.cached_kv = [(marker, marker + 100)]
            self.cached_freqs_gen = (marker + 200, marker + 300)
        return torch.full_like(hidden_states, float(token))


def passthrough_progress_bar(iterable):
    return iterable


@pytest.fixture(autouse=True)
def fake_cosmos3_guardrails(monkeypatch: pytest.MonkeyPatch):
    module = types.ModuleType("vllm_omni.diffusion.models.cosmos3.guardrails")
    module.is_guardrails_enabled = lambda od_config, sampling_params=None: False
    module.ensure_initialized = lambda od_config: None
    module.check_text_safety = lambda text: None
    module.check_video_safety = lambda video: video
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return module


@pytest.fixture
def make_cosmos3_pipeline():
    def _make():
        from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import (
            Cosmos3OmniDiffusersPipeline,
        )

        pipeline = object.__new__(Cosmos3OmniDiffusersPipeline)
        nn.Module.__init__(pipeline)
        pipeline.od_config = SimpleNamespace()
        pipeline.device = torch.device("cpu")
        pipeline.dtype = torch.float32
        pipeline.transformer = StubCosmos3Transformer(latent_channel_size=2)
        pipeline.vae = StubCosmos3VAE(z_dim=2)
        pipeline.vae_scale_factor_temporal = 4
        pipeline.vae_scale_factor_spatial = 8
        pipeline.scheduler = StubScheduler([9, 3], flow_shift=1.0)
        pipeline._base_scheduler_config = pipeline.scheduler.config
        pipeline._engine_init_flow_shift = 1.0
        pipeline._current_flow_shift = 1.0
        pipeline._guidance_scale = None
        pipeline._num_timesteps = None
        pipeline._cache_dit_requires_paired_cfg = False
        pipeline.progress_bar = passthrough_progress_bar
        return pipeline

    return _make


def make_sampling_params(**overrides: Any) -> SimpleNamespace:
    values = {
        "height": None,
        "width": None,
        "num_frames": None,
        "num_inference_steps": None,
        "guidance_scale": None,
        "generator": None,
        "seed": 123,
        "num_outputs_per_prompt": 1,
        "frame_rate": None,
        "resolved_frame_rate": None,
        "max_sequence_length": None,
        "extra_args": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _ids(value: int) -> torch.Tensor:
    return torch.tensor([[value]], dtype=torch.long)


def _mask() -> torch.Tensor:
    return torch.ones(1, 1, dtype=torch.long)


def test_pipeline_registered_and_exported() -> None:
    from vllm_omni.diffusion.cache.cache_dit_backend import CUSTOM_DIT_ENABLERS
    from vllm_omni.diffusion.models import cosmos3
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline
    from vllm_omni.diffusion.models.progress_bar import ProgressBarMixin
    from vllm_omni.diffusion.registry import (
        _DIFFUSION_MODELS,
        _DIFFUSION_POST_PROCESS_FUNCS,
        _DIFFUSION_PRE_PROCESS_FUNCS,
    )

    assert issubclass(Cosmos3OmniDiffusersPipeline, nn.Module)
    assert issubclass(Cosmos3OmniDiffusersPipeline, ProgressBarMixin)
    assert Cosmos3OmniDiffusersPipeline.support_image_input is True
    assert _DIFFUSION_MODELS["Cosmos3OmniDiffusersPipeline"] == (
        "cosmos3",
        "pipeline_cosmos3",
        "Cosmos3OmniDiffusersPipeline",
    )
    assert _DIFFUSION_PRE_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_pre_process_func"
    assert _DIFFUSION_POST_PROCESS_FUNCS["Cosmos3OmniDiffusersPipeline"] == "get_cosmos3_post_process_func"
    assert "Cosmos3OmniDiffusersPipeline" in CUSTOM_DIT_ENABLERS
    assert "Cosmos3OmniDiffusersPipeline" in cosmos3.__all__


def test_preprocess_i2v_image_input() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_pre_process_func

    preprocess = get_cosmos3_pre_process_func(SimpleNamespace())
    i2v = SimpleNamespace(
        prompts=[{"prompt": "A slow camera push.", "multi_modal_data": {"image": Image.new("RGB", (320, 160))}}],
        sampling_params=SimpleNamespace(height=None, width=None, extra_args={}),
    )

    result = preprocess(i2v)
    assert (result.sampling_params.height, result.sampling_params.width) == (672, 1344)
    assert tuple(result.prompts[0]["additional_information"]["preprocessed_image"].shape[-2:]) == (672, 1344)


def test_postprocess_handles_image_video_and_validation() -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import get_cosmos3_post_process_func

    func = get_cosmos3_post_process_func(SimpleNamespace())
    video = torch.zeros(1, 3, 1, 4, 4)

    assert func(video, output_type="latent") is video
    assert func({"image": video})[0].size == (4, 4)

    with pytest.raises(ValueError, match="text-to-image postprocess expects"):
        func({"image": torch.zeros(1, 3, 2, 4, 4)})
    with pytest.raises(ValueError, match="both image and video"):
        func({"image": video, "video": video})


def test_prompt_formatting_and_checkpoint_key_remap(make_cosmos3_pipeline) -> None:
    from vllm_omni.diffusion.models.cosmos3.pipeline_cosmos3 import Cosmos3OmniDiffusersPipeline

    pipeline = make_cosmos3_pipeline()
    captured: list[str] = []
    pipeline._tokenize_prompt = lambda text, *args, **kwargs: (captured.append(text) or _ids(len(captured)), _mask())

    pipeline._format_and_tokenize_prompts(
        "A robot",
        "bad",
        num_frames=48,
        frame_rate=24,
        height=720,
        width=1280,
        max_sequence_length=32,
        sp=SimpleNamespace(
            extra_args={
                "negative_metadata_mode": "inverse",
                # Duration/resolution metadata templates are off by default
                # (commit "Change resolution and duration templates to off by
                # default"); enable them explicitly to exercise the formatting.
                "use_duration_template": True,
                "use_resolution_template": True,
            }
        ),
        use_system_prompt=True,
        is_t2i=False,
    )
    assert "The video is 2.0 seconds long" in captured[0]
    assert "The video is not 2.0 seconds long" in captured[1]

    remaps = {
        "embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "model.embed_tokens.weight": "transformer.language_model.embed_tokens.weight",
        "norm.weight": "transformer.language_model.norm.weight",
        "norm_moe_gen.weight": "transformer.norm_moe_gen.weight",
        "proj_in.weight": "transformer.proj_in.weight",
        "proj_out.bias": "transformer.proj_out.bias",
        "layers.3.self_attn.to_q.weight": "transformer.language_model.layers.3.self_attn.to_q.weight",
        "layers.3.self_attn.to_out.weight": "transformer.language_model.layers.3.self_attn.to_out.weight",
        "layers.3.self_attn.norm_q.weight": "transformer.language_model.layers.3.self_attn.norm_q.weight",
        "layers.3.self_attn.add_q_proj.weight": "transformer.gen_layers.3.cross_attention.to_q.weight",
        "layers.3.self_attn.to_add_out.weight": "transformer.gen_layers.3.cross_attention.to_out.weight",
        "layers.3.self_attn.norm_added_q.weight": "transformer.gen_layers.3.cross_attention.norm_q.weight",
        "transformer.model.layers.3.self_attn.add_k_proj.weight": (
            "transformer.gen_layers.3.cross_attention.to_k.weight"
        ),
    }
    assert {key: Cosmos3OmniDiffusersPipeline._remap_ckpt_key(key) for key in remaps} == remaps


def test_prepare_latents_for_video_and_image(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = pipeline._prepare_latents(16, 24, 5, torch.Generator(device="cpu").manual_seed(0))
    assert latents.shape == (1, 2, 2, 2, 3)

    pipeline._encode_conditioning_video = lambda *args, **kwargs: torch.full((1, 2, 2, 2, 3), 5.0)
    i2v_latents, velocity_mask, image_latent = pipeline._prepare_latents_i2v(
        torch.zeros(1, 3, 16, 24), 16, 24, 5, torch.Generator(device="cpu").manual_seed(0)
    )
    torch.testing.assert_close(i2v_latents[:, :, 0], torch.full((1, 2, 2, 3), 5.0))
    assert velocity_mask.tolist() == [[[[[0.0]], [[1.0]]]]]
    assert image_latent.shape == (1, 2, 1, 2, 3)


def test_diffuse_covers_cfg_and_i2v_steps(make_cosmos3_pipeline) -> None:
    pipeline = make_cosmos3_pipeline()
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2]
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))

    i2v = pipeline.diffuse(
        latents=torch.zeros(1, 2, 2, 1, 1),
        timesteps=torch.tensor([7]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=1.0,
        shared_kwargs={"video_shape": (2, 1, 1), "fps": 24.0},
        velocity_mask=torch.tensor([[[[[0.0]], [[1.0]]]]]),
        image_latent=torch.full((1, 2, 1, 1, 1), 7.0),
    )
    torch.testing.assert_close(i2v[:, :, 0:1], torch.full((1, 2, 1, 1, 1), 7.0))


def test_diffuse_keeps_paired_cfg_when_cache_dit_active(make_cosmos3_pipeline) -> None:
    """With cache-dit active the uncond pass is kept even outside the guidance
    interval (so cache-dit's has_separate_cfg parity stays in phase), and the
    output is numerically identical to the skip path.

    Contrast with ``test_diffuse_covers_cfg_and_i2v_steps`` (no marker), where
    the same inputs skip the out-of-interval uncond pass: calls == [2, 1, 2].
    """
    pipeline = make_cosmos3_pipeline()
    # Marker normally set by ``enable_cache_for_cosmos3`` when cache-dit is on.
    pipeline._cache_dit_requires_paired_cfg = True
    latents = torch.zeros(1, 2, 1, 1, 1)

    result = pipeline.diffuse(
        latents=latents,
        timesteps=torch.tensor([900, 100]),
        cond_ids=_ids(2),
        cond_mask=_mask(),
        uncond_ids=_ids(1),
        uncond_mask=_mask(),
        guidance_scale=3.0,
        shared_kwargs={"video_shape": (1, 1, 1), "fps": 24.0},
        guidance_interval=(500.0, 1000.0),
    )

    # t=900 is inside the interval (cond+uncond); t=100 is outside but the
    # uncond pass is still issued -> paired cond/uncond at every step.
    assert [call["token"] for call in pipeline.transformer.calls] == [2, 1, 2, 1]
    # Identical result to the skip path: out-of-interval combine uses scale=1.0,
    # so combine_cfg_noise(cond=2, uncond=1, 1.0) == 2 == the skipped cond value.
    torch.testing.assert_close(result, torch.full_like(latents, 6.0))


class TestForwardRouting:
    def _install_forward_stubs(self, pipeline):
        captured: dict[str, object] = {"diffuse_calls": [], "prepare_calls": []}

        def fake_format(prompt, negative_prompt, num_frames, frame_rate, height, width, *args, **kwargs):
            captured["format"] = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "height": height,
                "width": width,
                "is_t2i": kwargs["is_t2i"],
            }
            return _ids(2), _mask(), _ids(1), _mask()

        def fake_prepare(height, width, num_frames, generator):
            captured["prepare_calls"].append((height, width, num_frames, generator.initial_seed()))
            return torch.zeros(1, 2, 1, 1, 1)

        def fake_diffuse(**kwargs):
            captured["diffuse_calls"].append(kwargs)
            return kwargs["latents"] + len(captured["diffuse_calls"])

        pipeline._format_and_tokenize_prompts = fake_format
        pipeline._prepare_latents = fake_prepare
        pipeline._set_flow_shift = lambda target: captured.setdefault("flow_shifts", []).append(target)
        pipeline.diffuse = fake_diffuse
        pipeline._decode_latents = lambda latents: latents
        return captured

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "expected"),
        [
            (
                {"prompt": "A painted robot", "modalities": ["image"]},
                make_sampling_params(num_outputs_per_prompt=2),
                {"key": "image", "is_t2i": True, "flow": [3.0], "steps": [50, 50], "frames": 1},
            ),
            (
                "A warehouse robot",
                make_sampling_params(),
                {"key": "video", "is_t2i": False, "flow": [1.0], "steps": [35], "frames": 189},
            ),
        ],
    )
    def test_forward_defaults_and_mode_selection(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        expected,
    ) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)

        output = pipeline.forward(SimpleNamespace(prompts=[prompt], sampling_params=sampling_params))

        assert expected["key"] in output.output
        assert captured["format"]["is_t2i"] is expected["is_t2i"]
        assert captured["format"]["num_frames"] == expected["frames"]
        assert captured["flow_shifts"] == expected["flow"]
        assert [call[0] for call in pipeline.scheduler.set_timesteps_calls] == expected["steps"]

    def test_forward_i2v_route(self, make_cosmos3_pipeline) -> None:
        pipeline = make_cosmos3_pipeline()
        captured = self._install_forward_stubs(pipeline)
        image_tensor = torch.zeros(1, 3, 16, 16)
        velocity_mask = torch.ones(1, 1, 1, 1, 1)

        pipeline._prepare_latents_i2v = lambda *args, **kwargs: (
            torch.zeros(1, 2, 1, 1, 1),
            velocity_mask,
            torch.zeros(1, 2, 1, 1, 1),
        )
        pipeline.forward(
            SimpleNamespace(
                prompts=[
                    {
                        "prompt": "move",
                        "modalities": ["video"],
                        "additional_information": {"preprocessed_image": image_tensor},
                    }
                ],
                sampling_params=make_sampling_params(height=16, width=16, num_frames=5),
            )
        )
        assert captured["diffuse_calls"][-1]["shared_kwargs"]["noisy_frame_mask"] is velocity_mask

    @pytest.mark.parametrize(
        ("prompt", "sampling_params", "message"),
        [
            (["one", "two"], make_sampling_params(), "single prompt"),
            ([{"prompt": "one", "modalities": ["image", "video"]}], make_sampling_params(), "both image and video"),
        ],
    )
    def test_forward_rejects_invalid_public_requests(
        self,
        make_cosmos3_pipeline,
        prompt,
        sampling_params,
        message,
    ) -> None:
        pipeline = make_cosmos3_pipeline()

        with pytest.raises(ValueError, match=message):
            pipeline.forward(SimpleNamespace(prompts=prompt, sampling_params=sampling_params))
