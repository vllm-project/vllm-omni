# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Exercise step selectors through both Qwen2.1 denoising entry points."""

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.forward_context import get_forward_context, set_forward_context
from vllm_omni.diffusion.models.qwen_image_21.cfg_parallel import QwenImage21CFGParallelMixin
from vllm_omni.diffusion.models.qwen_image_21.pipeline_qwen_image_21 import QwenImage21Pipeline
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


def _attention(config: OmniDiffusionConfig) -> Attention:
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer._kv_cache_dtype = config.diffusion_kv_cache_dtype
    layer._kv_cache_skip_steps = config.diffusion_kv_cache_skip_step_indices
    layer._kv_cache_skip_layers = None
    layer._disable_kv_quant = False
    return layer


class _Transformer:
    causal_condition = False
    transformer_blocks = ()

    def __init__(self) -> None:
        self.releases = 0

    def release_request_geometry(self) -> None:
        self.releases += 1


class _Scheduler:
    def set_begin_index(self, index: int) -> None:
        assert index == 0


class _Loop(QwenImage21CFGParallelMixin):
    def __init__(self, attention: Attention, fail_at: int | None = None) -> None:
        self.attention = attention
        self.transformer = _Transformer()
        self.scheduler = _Scheduler()
        self.interrupt = False
        self._progress_bar_config = {"disable": True}
        self.seen: list[tuple[int | None, str]] = []
        self.fail_at = fail_at

    def predict_noise(self, **kwargs: object) -> torch.Tensor:
        step = get_forward_context().denoise_step_idx
        metadata = self.attention._with_kv_cache_dtype(None)
        self.seen.append((step, metadata.extra["kv_cache_dtype"]))
        if step == self.fail_at and self.fail_at is not None:
            raise RuntimeError("denoise failed")
        hidden_states = kwargs["hidden_states"]
        assert isinstance(hidden_states, torch.Tensor)
        return hidden_states

    def scheduler_step_maybe_with_cfg(self, noise_pred, timestep, latents, do_true_cfg):
        return latents


def _diffuse(pipe: _Loop, do_true_cfg: bool = False) -> torch.Tensor:
    return pipe.diffuse(
        prompt_embeds=torch.zeros(1, 1, 4),
        prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool),
        negative_prompt_embeds=torch.zeros(1, 1, 4) if do_true_cfg else None,
        negative_prompt_embeds_mask=torch.ones(1, 1, dtype=torch.bool) if do_true_cfg else None,
        img_mask=torch.ones(1, 3, dtype=torch.bool),
        negative_img_mask=torch.ones(1, 3, dtype=torch.bool) if do_true_cfg else None,
        latents=torch.zeros(1, 2, 4),
        image_latents=None,
        img_shapes=[[(1, 1, 2)]],
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        do_true_cfg=do_true_cfg,
        true_cfg_scale=4.0 if do_true_cfg else 1.0,
    )


@pytest.mark.parametrize("do_true_cfg", [False, True])
def test_loop_publishes_step_indices_and_resets_between_requests(do_true_cfg):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _Loop(_attention(config))
    expected = [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")]
    if do_true_cfg:
        expected = [entry for entry in expected for _ in range(2)]
    with set_forward_context(omni_diffusion_config=config):
        for _ in range(2):
            pipe.seen.clear()
            _diffuse(pipe, do_true_cfg)
            assert pipe.seen == expected
            assert get_forward_context().denoise_step_idx is None
    assert pipe.transformer.releases == 2


def test_loop_clears_step_context_after_failure():
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _Loop(_attention(config), fail_at=1)
    with set_forward_context(omni_diffusion_config=config):
        with pytest.raises(RuntimeError, match="denoise failed"):
            _diffuse(pipe)
        assert get_forward_context().denoise_step_idx is None
    assert pipe.transformer.releases == 1


def _state(request_id: str, step: int, fill: float, rows: int = 1) -> StepRequestState:
    return StepRequestState(
        request_id=request_id,
        sampling=OmniDiffusionSamplingParams(true_cfg_scale=1.0),
        prompt_embeds=torch.ones(rows, 1, 4),
        prompt_embeds_mask=torch.ones(rows, 1, dtype=torch.bool),
        latents=torch.full((rows, 2, 4), fill),
        timesteps=torch.tensor([999.0, 700.0, 200.0, 1.0]),
        step_index=step,
        img_shapes=[[(1, 1, 2)]] * rows,
        extra={"image_pad_mask": torch.ones(rows, 1, dtype=torch.bool), "image_latents": None, "kv_cache": None},
    )


def _step_pipeline(config: OmniDiffusionConfig) -> QwenImage21Pipeline:
    pipe = QwenImage21Pipeline.__new__(QwenImage21Pipeline)
    nn.Module.__init__(pipe)
    pipe.od_config = config
    pipe.transformer = nn.Module()
    pipe._interrupt = False
    pipe._attention_kwargs = {}
    return pipe


@pytest.mark.parametrize("step,expected", [(0, "float"), (1, "mxfp8"), (2, "float"), (3, "mxfp8")])
def test_step_mode_publishes_request_step(monkeypatch, step, expected):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0,2")
    pipe = _step_pipeline(config)
    attention = _attention(config)
    seen = []

    def predict(*args, **kwargs):
        seen.append(
            (get_forward_context().denoise_step_idx, attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"])
        )
        return args[2]["hidden_states"]

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch([_state("a", step, 1.0)]))
    assert result.shape == (1, 2, 4)
    assert seen == [(step, expected)]


@pytest.mark.parametrize("skip_steps", [None, "0,2"])
def test_step_mode_separates_quantization_steps_without_reordering_rows(monkeypatch, skip_steps):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps=skip_steps)
    pipe = _step_pipeline(config)
    attention = _attention(config)
    states = [_state("a", 1, 1.0), _state("b", 2, 2.0, rows=2), _state("c", 1, 3.0)]
    seen = []

    def predict(*args, **kwargs):
        hidden = args[2]["hidden_states"]
        seen.append(
            (
                get_forward_context().denoise_step_idx,
                attention._with_kv_cache_dtype(None).extra["kv_cache_dtype"],
                hidden.shape[0],
            )
        )
        return hidden

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        result = pipe.denoise_step(InputBatch.make_batch(states))
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1.0, 2.0, 2.0, 3.0]))
    expected: list[tuple[int | None, str, int]]
    if skip_steps:
        expected = [(1, "mxfp8", 2), (2, "float", 2)]
    else:
        expected = [(None, "mxfp8", 4)]
    assert seen == expected


def test_step_mode_clears_context_after_failure(monkeypatch):
    config = OmniDiffusionConfig(diffusion_kv_cache_dtype="mxfp8", diffusion_kv_cache_skip_steps="0")
    pipe = _step_pipeline(config)

    def predict(*args, **kwargs):
        assert get_forward_context().denoise_step_idx == 1
        raise RuntimeError("denoise failed")

    monkeypatch.setattr(pipe, "predict_noise_maybe_with_cfg", predict)
    with set_forward_context(omni_diffusion_config=config):
        with pytest.raises(RuntimeError, match="denoise failed"):
            pipe.denoise_step(InputBatch.make_batch([_state("a", 1, 1.0)]))
        assert get_forward_context().denoise_step_idx is None
