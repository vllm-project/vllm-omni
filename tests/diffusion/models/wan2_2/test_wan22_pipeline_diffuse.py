# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.wan2_2.pipeline_wan2_2 import Wan22Pipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _StubTransformer(nn.Module):
    @property
    def dtype(self) -> torch.dtype:
        return torch.float32


class _StubScheduler:
    def __init__(self, timesteps: list[int]) -> None:
        self.timesteps = torch.tensor(timesteps, dtype=torch.int64)
        self.config = SimpleNamespace(num_train_timesteps=1000)
        self.set_timesteps_calls: list[tuple[int, torch.device]] = []

    def set_timesteps(self, num_steps: int, device: torch.device) -> None:
        self.set_timesteps_calls.append((num_steps, device))


@contextmanager
def _noop_progress_bar(*args, **kwargs):
    del args, kwargs

    class _Bar:
        def update(self) -> None:
            return None

    yield _Bar()


def _make_pipeline() -> Wan22Pipeline:
    pipeline = object.__new__(Wan22Pipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.transformer = _StubTransformer()
    pipeline.transformer_2 = None
    pipeline.transformer_config = SimpleNamespace(patch_size=(1, 2, 2), in_channels=4, out_channels=4)
    pipeline.scheduler = _StubScheduler([9, 5])
    pipeline.od_config = SimpleNamespace(flow_shift=5.0)
    pipeline._sample_solver = "unipc"
    pipeline._flow_shift = 5.0
    pipeline.vae_scale_factor_temporal = 4
    pipeline.vae_scale_factor_spatial = 8
    pipeline.boundary_ratio = 0.875
    pipeline.expand_timesteps = False
    pipeline._guidance_scale = None
    pipeline._guidance_scale_2 = None
    pipeline._num_timesteps = None
    pipeline._current_timestep = None
    pipeline.check_inputs = lambda **kwargs: None
    pipeline.prepare_latents = lambda **kwargs: torch.zeros((1, 4, 1, 8, 8), dtype=torch.float32)
    pipeline.progress_bar = _noop_progress_bar
    return pipeline


def test_forward_delegates_denoising_to_diffuse(monkeypatch) -> None:
    pipeline = _make_pipeline()

    prompt_embeds = torch.randn(1, 8)
    captured: dict[str, object] = {}

    def _fake_diffuse(**kwargs):
        captured.update(kwargs)
        return kwargs["latents"] + 1

    pipeline.diffuse = _fake_diffuse  # type: ignore[method-assign]

    req = SimpleNamespace(
        prompts=["prompt"],
        sampling_params=SimpleNamespace(
            height=None,
            width=None,
            num_frames=1,
            num_inference_steps=2,
            guidance_scale_provided=False,
            guidance_scale=None,
            guidance_scale_2=None,
            boundary_ratio=None,
            generator=None,
            seed=None,
            num_outputs_per_prompt=1,
            max_sequence_length=32,
            latents=None,
            extra_args={},
        ),
    )

    output = pipeline.forward(req, prompt_embeds=prompt_embeds, output_type="latent", guidance_scale=1.0)

    assert torch.equal(output.output, torch.ones((1, 4, 1, 8, 8)))
    assert torch.equal(captured["timesteps"], pipeline.scheduler.timesteps)
    assert captured["guidance_low"] == 1.0
    assert captured["guidance_high"] == 1.0
    assert captured["boundary_timestep"] == pytest.approx(875.0)
    assert captured["latent_condition"] is None
    assert captured["first_frame_mask"] is None
    assert pipeline.scheduler.set_timesteps_calls == [(2, torch.device("cpu"))]


def test_diffuse_runs_prediction_and_scheduler_for_each_timestep() -> None:
    pipeline = _make_pipeline()
    latents = torch.zeros((1, 1, 1, 2, 2), dtype=torch.float32)
    timesteps = torch.tensor([7, 3], dtype=torch.int64)
    prompt_embeds = torch.randn(1, 8)

    predict_calls: list[dict[str, object]] = []
    scheduler_calls: list[tuple[float, int, float, bool]] = []

    def _fake_predict_noise_maybe_with_cfg(**kwargs):
        predict_calls.append(kwargs)
        timestep = kwargs["positive_kwargs"]["timestep"]
        assert isinstance(timestep, torch.Tensor)
        return torch.full_like(latents, float(timestep[0].item()))

    def _fake_scheduler_step_maybe_with_cfg(noise_pred, t, current_latents, do_true_cfg):
        scheduler_calls.append(
            (float(noise_pred[0, 0, 0, 0, 0]), int(t.item()), float(current_latents.sum()), do_true_cfg)
        )
        return current_latents + noise_pred

    pipeline.predict_noise_maybe_with_cfg = _fake_predict_noise_maybe_with_cfg  # type: ignore[method-assign]
    pipeline.scheduler_step_maybe_with_cfg = _fake_scheduler_step_maybe_with_cfg  # type: ignore[method-assign]

    result = pipeline.diffuse(
        latents=latents,
        timesteps=timesteps,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=None,
        guidance_low=1.0,
        guidance_high=2.0,
        boundary_timestep=5.0,
        dtype=torch.float32,
        attention_kwargs={},
    )

    assert len(predict_calls) == 2
    assert predict_calls[0]["true_cfg_scale"] == 1.0
    assert predict_calls[1]["true_cfg_scale"] == 2.0
    assert scheduler_calls == [
        (7.0, 7, 0.0, False),
        (3.0, 3, 28.0, False),
    ]
    assert torch.equal(result, torch.full_like(latents, 10.0))
