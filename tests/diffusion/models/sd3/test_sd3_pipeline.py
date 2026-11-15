# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.sd3.pipeline_sd3 import StableDiffusion3Pipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

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
    pipeline.vae_scale_factor = 8
    pipeline.patch_size = 2
    pipeline.default_sample_size = 128
    pipeline.transformer = SimpleNamespace(in_channels=1)
    return pipeline


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
