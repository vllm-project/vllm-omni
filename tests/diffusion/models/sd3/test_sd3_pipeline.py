# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import copy
from types import SimpleNamespace

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from vllm_omni.diffusion.data import OmniDiffusionConfig
from vllm_omni.diffusion.models.interface import supports_step_execution
from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_sd3_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "max_sequence_length": None,
        "num_outputs_per_prompt": 0,
        "generator": None,
        "latents": None,
        "guidance_scale": 4.0,
        "output_type": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_sd3_pipeline():
    pipeline = object.__new__(StableDiffusion3Pipeline)
    nn.Module.__init__(pipeline)
    pipeline._interrupt = False
    pipeline.vae_scale_factor = 8
    pipeline.patch_size = 2
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(in_channels=1)
    return pipeline


def test_sd3_algorithm_adapter_can_disable_inherited_step_execution():
    pipeline = _make_sd3_pipeline()
    pipeline.supports_step_execution = False
    assert not supports_step_execution(pipeline)


def test_forward_collates_request_prompt_tensors_for_sd3():
    pipeline = _make_sd3_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    encode_calls = []
    diffuse_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_calls.append(kwargs)
        prompt_embeds = kwargs["prompt_embeds"]
        if prompt_embeds is None:
            prompt_embeds = torch.empty(2, 2, 3)
        return prompt_embeds, kwargs.get("pooled_prompt_embeds")

    def _fake_diffuse(**kwargs):
        diffuse_call.update(kwargs)
        raise StopAfterDiffuseError

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(2, 1, 1, 1)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.diffuse = _fake_diffuse

    prompt_embeds_a = torch.zeros(2, 3)
    prompt_embeds_b = torch.ones(2, 3)
    pooled_prompt_embeds_a = torch.full((4,), 2.0)
    pooled_prompt_embeds_b = torch.full((4,), 3.0)
    negative_prompt_embeds_a = torch.full((2, 3), 4.0)
    negative_prompt_embeds_b = torch.full((2, 3), 5.0)
    negative_pooled_prompt_embeds_a = torch.full((4,), 6.0)
    negative_pooled_prompt_embeds_b = torch.full((4,), 7.0)

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="sd3-prompt-a",
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "pooled_prompt_embeds": pooled_prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds_a,
                },
                sampling_params=_make_sd3_sampling(),
            ),
            SimpleNamespace(
                request_id="sd3-prompt-b",
                prompt={
                    "prompt": "prompt-b",
                    "negative_prompt": "negative-b",
                    "additional_information": {
                        "prompt_embeds": [prompt_embeds_b],
                        "pooled_prompt_embeds": [pooled_prompt_embeds_b],
                        "negative_prompt_embeds": [negative_prompt_embeds_b],
                        "negative_pooled_prompt_embeds": [negative_pooled_prompt_embeds_b],
                    },
                },
                sampling_params=_make_sd3_sampling(),
            ),
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert encode_calls[0]["prompt"] is None
    assert encode_calls[0]["prompt_2"] is None
    assert encode_calls[0]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[0]["prompt_embeds"],
        torch.stack([prompt_embeds_a, prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[0]["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )

    assert encode_calls[1]["prompt"] is None
    assert encode_calls[1]["prompt_2"] is None
    assert encode_calls[1]["prompt_3"] is None
    torch.testing.assert_close(
        encode_calls[1]["prompt_embeds"],
        torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[1]["pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["pooled_prompt_embeds"],
        torch.stack([pooled_prompt_embeds_a, pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["negative_pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )


def test_encode_prompt_preserves_direct_pooled_prompt_embeds():
    pipeline = _make_sd3_pipeline()
    prompt_embeds = torch.zeros(1, 2, 3)
    pooled_prompt_embeds = torch.ones(1, 4)

    actual_prompt_embeds, actual_pooled_prompt_embeds = pipeline.encode_prompt(
        prompt=None,
        prompt_2=None,
        prompt_3=None,
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
    )

    assert actual_prompt_embeds is prompt_embeds
    assert actual_pooled_prompt_embeds is pooled_prompt_embeds


@pytest.mark.parametrize(
    ("guidance_scale", "expected"),
    [
        pytest.param(4.0, True, id="cfg"),
        pytest.param(1.0, False, id="no-cfg"),
    ],
)
def test_forward_tells_cache_dit_whether_the_request_runs_cfg(
    monkeypatch: pytest.MonkeyPatch,
    guidance_scale: float,
    expected: bool,
):
    """Regression test for https://github.com/vllm-project/vllm-omni/issues/8734."""
    pipeline = _make_sd3_pipeline()

    class StopAfterDiffuseError(Exception):
        pass

    def _fake_diffuse(**kwargs):
        raise StopAfterDiffuseError

    pipeline.encode_prompt = lambda **kwargs: (torch.empty(1, 2, 3), torch.empty(1, 4))
    pipeline.prepare_latents = lambda *args, **kwargs: torch.zeros(1, 1, 1, 1)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0]), 1)
    pipeline.diffuse = _fake_diffuse

    calls = []
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.sd3.pipeline_sd3.set_separate_cfg",
        lambda transformer, do_true_cfg: calls.append((transformer, do_true_cfg)),
    )

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="sd3-cache-dit-cfg",
                prompt="a prompt",
                sampling_params=_make_sd3_sampling(guidance_scale=guidance_scale),
            )
        ]
    )

    with pytest.raises(StopAfterDiffuseError):
        pipeline.forward(batch)

    assert calls == [(pipeline.transformer, expected)]


def _make_step_sd3_pipeline():
    pipeline = _make_sd3_pipeline()
    pipeline.device = torch.device("cpu")
    pipeline.od_config = OmniDiffusionConfig(dtype=torch.float32)
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler()
    pipeline.encode_prompt = lambda **values: (values["prompt_embeds"], values["pooled_prompt_embeds"])

    def predict(do_cfg, scale, positive, negative, normalize):
        def branch(values):
            condition = values["encoder_hidden_states"].mean(dim=(1, 2))
            pooled = values["pooled_projections"].mean(dim=1)
            timestep = values["timestep"].reshape(-1, 1, 1, 1)
            return values["hidden_states"] * 0.1 + timestep * 0.001 + (condition + pooled).reshape(-1, 1, 1, 1)

        conditional = branch(positive)
        if not do_cfg:
            return conditional
        unconditional = branch(negative)
        return unconditional + scale * (conditional - unconditional)

    pipeline.predict_noise_maybe_with_cfg = predict
    return pipeline


def _make_step_sd3_requests(guidance_scale):
    requests = []
    for index in range(2):
        sampling = OmniDiffusionSamplingParams(
            height=16,
            width=16,
            num_inference_steps=3,
            guidance_scale=guidance_scale,
            output_type="latent",
            latents=torch.full((1, 1, 2, 2), float(index)),
        )
        requests.append(
            OmniDiffusionRequest(
                request_id=f"step-{index}",
                sampling_params=sampling,
                prompt={
                    "prompt_embeds": torch.full((2, 3), float(index + 1)),
                    "pooled_prompt_embeds": torch.full((4,), float(index + 2)),
                    "negative_prompt_embeds": torch.full((2, 3), -float(index + 1)),
                    "negative_pooled_prompt_embeds": torch.full((4,), -float(index + 2)),
                },
            )
        )
    return requests


@pytest.mark.parametrize("guidance_scale", [1.0, 4.0])
def test_step_execution_matches_full_loop_with_reordered_requests(monkeypatch, guidance_scale):
    pipeline = _make_step_sd3_pipeline()
    monkeypatch.setattr("vllm_omni.diffusion.models.sd3.pipeline_sd3.set_separate_cfg", lambda *args: None)
    requests = _make_step_sd3_requests(guidance_scale)
    expected = pipeline.forward(DiffusionRequestBatch(requests=copy.deepcopy(requests)))
    states = [
        StepRequestState(request_id=req.request_id, sampling=req.sampling_params, prompt=req.prompt)
        for req in copy.deepcopy(requests)
    ]

    def advance(step_states):
        order = torch.arange(len(step_states) - 1, -1, -1)
        batch = InputBatch.make_batch(step_states, idx_mapping=order)
        predictions = pipeline.denoise_step(batch)
        for state, prediction in zip(batch.states, predictions.split(1), strict=True):
            pipeline.step_scheduler(state, prediction)

    pipeline.prepare_encode(states[0])
    advance(states[:1])
    pipeline.prepare_encode(states[1])
    assert states[0].scheduler is not states[1].scheduler
    assert all(state.scheduler is not pipeline.scheduler for state in states)
    for _ in range(2):
        advance(states)
    advance(states[1:])
    for state, output in zip(states, expected, strict=True):
        assert state.denoise_completed and state.step_index == 3
        torch.testing.assert_close(pipeline.post_decode(state).output, output.output, rtol=0, atol=0)
