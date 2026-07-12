from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as mammoth_pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline

pytestmark = [pytest.mark.cpu]


class _FakeScheduler:
    def set_timesteps(self, num_inference_steps, device, num_tokens):  # noqa: ARG002
        self.timesteps = torch.arange(num_inference_steps, 0, -1, device=device, dtype=torch.float32)

    def step(self, model_pred, t, latents, return_dict=False):  # noqa: ARG002
        return (latents,)


class _FakeTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(in_channels=4)
        self.param = nn.Parameter(torch.zeros(()))
        self.branches: list[str | None] = []

    def forward(self, hidden_states, **kwargs):
        self.branches.append(kwargs.get("teacache_branch"))
        return torch.zeros_like(hidden_states)


class _FakeVAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(scaling_factor=None, shift_factor=None)

    def decode(self, latents, return_dict=False):  # noqa: ARG002
        return (latents,)


def _build_pipeline(monkeypatch):
    monkeypatch.setattr(mammoth_pipeline_module, "FlowMatchEulerDiscreteScheduler", _FakeScheduler)

    pipe = MammothModa2DiTPipeline.__new__(MammothModa2DiTPipeline)
    nn.Module.__init__(pipe)
    pipe.gen_transformer = _FakeTransformer()
    pipe.gen_vae = _FakeVAE()
    pipe.gen_image_condition_refiner = None
    pipe.gen_freqs_cis = []
    pipe._llm_hidden_size = 8
    return pipe


def _runtime_info():
    return {
        "text_prompt_embeds": torch.randn(2, 8),
        "image_prompt_embeds": torch.randn(1, 8),
        "negative_prompt_embeds": torch.randn(1, 8),
        "negative_prompt_attention_mask": [True],
        "image_height": [32],
        "image_width": [32],
        "text_guidance_scale": [1.0],
        "cfg_range": [0.0, 1.0],
        "num_inference_steps": [4],
    }


def _run_pipeline(pipe, *, text_guidance_scale, cfg_range, num_inference_steps=4):
    pipe(
        inputs_embeds=torch.zeros(1, 8),
        runtime_additional_information=[_runtime_info()],
        sampling_extra_args=[
            {
                "text_guidance_scale": text_guidance_scale,
                "cfg_range": cfg_range,
                "num_inference_steps": num_inference_steps,
            }
        ],
    )
    return pipe.gen_transformer.branches


def test_mammoth_moda2_non_cfg_passes_positive_teacache_branch(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=1.0, cfg_range=[0.0, 1.0])

    assert branches == ["positive", "positive", "positive", "positive"]


def test_mammoth_moda2_cfg_passes_positive_then_negative_teacache_branch(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=4.0, cfg_range=[0.0, 1.0])

    assert branches == [
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
        "positive",
        "negative",
    ]


def test_mammoth_moda2_cfg_range_only_uses_negative_inside_range(monkeypatch):
    pipe = _build_pipeline(monkeypatch)

    branches = _run_pipeline(pipe, text_guidance_scale=4.0, cfg_range=[0.5, 1.0])

    assert branches == [
        "positive",
        "positive",
        "positive",
        "negative",
        "positive",
        "negative",
    ]
