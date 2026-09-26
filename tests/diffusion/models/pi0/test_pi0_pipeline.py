# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU tests for the Pi0 serving request contract."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from vllm_omni.diffusion.models.pi.pi0 import pipeline_pi0
from vllm_omni.diffusion.models.pi.pi0.config import Pi0Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _SpyModel:
    def __init__(self, output_dtype=torch.float32):
        self.seen_steps = []
        self.output_dtype = output_dtype

    @staticmethod
    def _normalize_state(state):
        return state

    @staticmethod
    def _unnormalize_actions(actions):
        return actions

    def sample_actions(
        self,
        *,
        images,
        image_masks,
        lang_tokens,
        lang_masks,
        state,
        noise,
        num_steps,
    ):
        del images, image_masks, lang_tokens, lang_masks, state, noise
        self.seen_steps.append(num_steps)
        return torch.zeros(1, 2, 3, dtype=self.output_dtype)


class _TinyPi0Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.paligemma_with_expert = torch.nn.Linear(4, 4)
        self.state_proj = torch.nn.Linear(3, 4)
        self.register_buffer("floating_buffer", torch.ones(1))


def _pipeline(monkeypatch, spy):
    pipeline = object.__new__(pipeline_pi0.Pi0Pipeline)
    pipeline.config = Pi0Config(chunk_size=2, max_action_dim=3, max_state_dim=3)
    pipeline.tokenizer = object()
    pipeline._device = torch.device("cpu")
    pipeline.model = spy
    monkeypatch.setattr(
        pipeline_pi0,
        "build_model_inputs",
        lambda robot_obs, config, tokenizer, device: (
            [torch.empty(0)],
            [torch.empty(0)],
            torch.empty(0),
            torch.empty(0),
            torch.zeros(1, 3),
        ),
    )
    return pipeline


def _request(top_level_steps, *, legacy_extra_steps=99):
    sampling_params = SimpleNamespace(
        num_inference_steps=top_level_steps,
        extra_args={
            "robot_obs": {},
            "num_inference_steps": legacy_extra_steps,
        },
    )
    return SimpleNamespace(sampling_params=sampling_params)


@pytest.mark.parametrize("requested", [None, 2, np.int64(25)])
def test_top_level_num_inference_steps_is_the_single_source(monkeypatch, requested):
    spy = _SpyModel()

    output = _pipeline(monkeypatch, spy).forward(_request(requested))

    expected = None if requested is None else int(requested)
    assert spy.seen_steps == [expected]
    assert output.output["actions"].shape == (2, 3)


@pytest.mark.parametrize("bad", [0, -1, 2.5, True, "4"])
def test_invalid_top_level_num_inference_steps_never_reaches_model(monkeypatch, bad):
    spy = _SpyModel()

    with pytest.raises(ValueError, match="num_inference_steps must be a positive integer"):
        _pipeline(monkeypatch, spy).forward(_request(bad))

    assert spy.seen_steps == []


def test_bfloat16_model_output_is_returned_as_float32(monkeypatch):
    output = _pipeline(monkeypatch, _SpyModel(torch.bfloat16)).forward(_request(2))

    assert output.output["actions"].dtype == np.float32


@pytest.mark.parametrize(
    "configured,expected",
    [
        ("float32", torch.float32),
        ("bfloat16", torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.bfloat16, torch.bfloat16),
    ],
)
def test_pipeline_resolves_supported_dtypes(configured, expected):
    assert pipeline_pi0.Pi0Pipeline._resolve_dtype(SimpleNamespace(dtype=configured)) is expected


@pytest.mark.parametrize("configured", ["float16", "float64", "auto", None])
def test_pipeline_rejects_unsupported_dtypes(configured):
    with pytest.raises(ValueError, match="Unsupported .* dtype"):
        pipeline_pi0.Pi0Pipeline._resolve_dtype(SimpleNamespace(dtype=configured))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pi0_inference_layout_is_homogeneous(dtype):
    model = _TinyPi0Model()

    pipeline_pi0._set_inference_dtype(model, dtype)

    assert {parameter.dtype for parameter in model.parameters()} == {dtype}
    assert {buffer.dtype for buffer in model.buffers() if buffer.is_floating_point()} == {dtype}


def test_bfloat16_layout_is_applied_before_checkpoint_load(monkeypatch, tmp_path):
    model = _TinyPi0Model()
    observed_dtypes = {}

    def fake_load_checkpoint(_self, model_to_load):
        observed_dtypes.update({name: parameter.dtype for name, parameter in model_to_load.named_parameters()})

    (tmp_path / "model.safetensors").touch()
    pipeline = object.__new__(pipeline_pi0.Pi0Pipeline)
    pipeline.model_dir = str(tmp_path)
    pipeline.config = object()
    pipeline._device = torch.device("cpu")
    pipeline._torch_dtype = torch.bfloat16
    monkeypatch.setattr(pipeline_pi0, "Pi0ForActionPrediction", lambda _config: model)
    monkeypatch.setattr(pipeline_pi0.Pi0Pipeline, "_load_checkpoint", fake_load_checkpoint)

    initialized = pipeline._initialize_model()

    assert initialized is model
    assert set(observed_dtypes.values()) == {torch.bfloat16}
    assert not initialized.training
