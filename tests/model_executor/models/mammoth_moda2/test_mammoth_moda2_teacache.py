from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.mammoth_moda2 import pipeline_mammothmoda2_dit as mammoth_pipeline_module
from vllm_omni.diffusion.models.mammoth_moda2.pipeline_mammothmoda2_dit import MammothModa2DiTPipeline
from vllm_omni.model_executor.models.mammoth_moda2 import mammoth_moda2 as mammoth_model_module
from vllm_omni.model_executor.models.mammoth_moda2.mammoth_moda2 import MammothModa2ForConditionalGeneration

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
    pipe.cache_backend = None
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


def test_mammoth_moda2_dit_stage_enables_cache_backend(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    fake_backend = SimpleNamespace(
        enable=Mock(),
        is_enabled=Mock(return_value=True),
        refresh=Mock(),
    )
    monkeypatch.setattr(
        mammoth_model_module,
        "get_cache_backend",
        lambda *_args, **_kwargs: fake_backend,
    )
    wrapper = object.__new__(MammothModa2ForConditionalGeneration)
    nn.Module.__init__(wrapper)
    wrapper.model_stage = "dit"
    wrapper.dit = pipe
    wrapper.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            cache_backend="tea_cache",
            cache_config={"rel_l1_thresh": 0.1},
        )
    )

    wrapper._maybe_enable_dit_cache_backend()

    fake_backend.enable.assert_called_once_with(pipe)
    assert wrapper._dit_cache_backend is fake_backend
    assert pipe.cache_backend is fake_backend


def test_mammoth_moda2_dit_stage_refreshes_cache_from_sampling_steps(monkeypatch):
    pipe = _build_pipeline(monkeypatch)
    fake_backend = SimpleNamespace(
        is_enabled=Mock(return_value=True),
        refresh=Mock(),
    )
    pipe.cache_backend = fake_backend

    _run_pipeline(pipe, text_guidance_scale=1.0, cfg_range=[0.0, 1.0], num_inference_steps=3)

    fake_backend.refresh.assert_called_once_with(pipe, 3, verbose=False)
