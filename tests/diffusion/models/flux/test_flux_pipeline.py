from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.flux.pipeline_flux import FluxPipeline
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _make_flux_sampling(**overrides):
    values = {
        "height": 32,
        "width": 32,
        "num_inference_steps": 2,
        "sigmas": None,
        "guidance_scale": 3.5,
        "generator": None,
        "true_cfg_scale": 4.0,
        "num_outputs_per_prompt": 0,
        "latents": None,
        "output_type": None,
        "max_sequence_length": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_flux_pipeline():
    pipeline = object.__new__(FluxPipeline)
    nn.Module.__init__(pipeline)
    pipeline.default_sample_size = 128
    pipeline.vae_scale_factor = 8
    pipeline.device = torch.device("cpu")
    pipeline.text_encoder = None
    pipeline.transformer = SimpleNamespace(
        in_channels=4,
        guidance_embeds=False,
        dtype=torch.float32,
    )
    return pipeline


def test_forward_collates_request_prompt_tensors_for_flux(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.diffusion.models.flux.pipeline_flux.get_classifier_free_guidance_world_size",
        lambda: 1,
    )
    pipeline = _make_flux_pipeline()
    encode_calls = []
    prepare_latents_call = {}
    diffuse_call = {}

    def _fake_encode_prompt(**kwargs):
        encode_calls.append(kwargs)
        return (
            kwargs["prompt_embeds"],
            kwargs["pooled_prompt_embeds"],
            torch.zeros(kwargs["prompt_embeds"].shape[1], 3),
        )

    def _fake_prepare_latents(
        batch_size,
        num_channels_latents,
        height,
        width,
        dtype,
        device,
        generator,
        latents,
    ):
        prepare_latents_call.update(
            {
                "batch_size": batch_size,
                "num_channels_latents": num_channels_latents,
                "height": height,
                "width": width,
                "dtype": dtype,
                "device": device,
                "generator": generator,
                "latents": latents,
            }
        )
        return torch.zeros(batch_size, 1, 1), torch.zeros(1, 3)

    def _fake_diffuse(
        prompt_embeds,
        pooled_prompt_embeds,
        negative_prompt_embeds,
        negative_pooled_prompt_embeds,
        *args,
        **kwargs,
    ):
        diffuse_call.update(
            {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "negative_prompt_embeds": negative_prompt_embeds,
                "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds,
            }
        )
        return torch.arange(2, dtype=torch.float32).view(2, 1)

    pipeline.encode_prompt = _fake_encode_prompt
    pipeline.prepare_latents = _fake_prepare_latents
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
    latents_a = torch.zeros(1, 1, 1)
    latents_b = torch.ones(1, 1, 1)
    gen_a = torch.Generator(device="cpu").manual_seed(1)
    gen_b = torch.Generator(device="cpu").manual_seed(2)

    batch = DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="flux-prompt-a",
                prompt={
                    "prompt": "prompt-a",
                    "negative_prompt": "negative-a",
                    "prompt_embeds": prompt_embeds_a,
                    "pooled_prompt_embeds": pooled_prompt_embeds_a,
                    "negative_prompt_embeds": negative_prompt_embeds_a,
                    "negative_pooled_prompt_embeds": negative_pooled_prompt_embeds_a,
                },
                sampling_params=_make_flux_sampling(generator=gen_a, latents=latents_a, output_type="latent"),
            ),
            SimpleNamespace(
                request_id="flux-prompt-b",
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
                sampling_params=_make_flux_sampling(generator=gen_b, latents=latents_b, output_type="latent"),
            ),
        ]
    )

    outputs = pipeline.forward(batch)

    assert encode_calls[0]["prompt"] is None
    assert encode_calls[0]["prompt_2"] is None
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
    torch.testing.assert_close(
        encode_calls[1]["prompt_embeds"],
        torch.stack([negative_prompt_embeds_a, negative_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        encode_calls[1]["pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )
    torch.testing.assert_close(
        diffuse_call["negative_pooled_prompt_embeds"],
        torch.stack([negative_pooled_prompt_embeds_a, negative_pooled_prompt_embeds_b], dim=0),
    )
    assert prepare_latents_call["generator"] == [gen_a, gen_b]
    torch.testing.assert_close(prepare_latents_call["latents"], torch.cat([latents_a, latents_b], dim=0))
    torch.testing.assert_close(outputs[0].output, torch.tensor([[0.0]]))
    torch.testing.assert_close(outputs[1].output, torch.tensor([[1.0]]))
