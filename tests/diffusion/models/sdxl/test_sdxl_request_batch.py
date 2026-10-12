# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Request-level batching contract for the SDXL text2image pipeline."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.data import DiffusionOutput
from vllm_omni.diffusion.models.sdxl.pipeline_sdxl import StableDiffusionXLPipeline
from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

HEIGHT = 64
WIDTH = 64
LATENT_CHANNELS = 4
VAE_SCALE = 8


class _StubScheduler:
    def __init__(self) -> None:
        self.timesteps = torch.tensor([9, 5], dtype=torch.int64)
        self.sigmas = torch.tensor([1.0, 0.5, 0.0])
        self.init_noise_sigma = 1.0
        self.set_timesteps_calls: list[int] = []

    def set_timesteps(self, num_steps: int) -> None:
        self.set_timesteps_calls.append(num_steps)


class _StubVAE:
    dtype = torch.float32
    config = SimpleNamespace(scaling_factor=0.13025)

    def decode(self, latents: torch.Tensor, return_dict: bool = True):
        # Emit one distinguishable image row per latent row so that the
        # per-request split can be asserted.
        batch = latents.shape[0]
        image = torch.stack([torch.full((3, HEIGHT, WIDTH), float(i)) for i in range(batch)])
        return (image,)


def _make_pipeline(recorder: dict) -> StableDiffusionXLPipeline:
    pipeline = object.__new__(StableDiffusionXLPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.od_config = SimpleNamespace(dtype=torch.float32)
    pipeline.vae_scale_factor = VAE_SCALE
    pipeline.default_sample_size = HEIGHT // VAE_SCALE
    pipeline.tokenizer_max_length = 77
    pipeline.output_type = "pt"
    pipeline.unet = SimpleNamespace(in_channels=LATENT_CHANNELS)
    pipeline.scheduler = _StubScheduler()
    pipeline.vae = _StubVAE()

    def _encode_prompt(prompt, num_images_per_prompt=1):
        recorder.setdefault("encoded_prompts", []).append(prompt)
        n = (1 if isinstance(prompt, str) else len(prompt)) * num_images_per_prompt
        return torch.zeros(n, 77, 2048), torch.zeros(n, 1280)

    def _diffuse(latents, **kwargs):
        recorder["diffuse_batch"] = latents.shape[0]
        recorder["diffuse_latents"] = latents.clone()
        return latents

    pipeline.encode_prompt = _encode_prompt
    pipeline.diffuse = _diffuse
    return pipeline


def _make_batch(prompts: list[str], latents: list[torch.Tensor] | None = None) -> DiffusionRequestBatch:
    requests = []
    for idx, prompt in enumerate(prompts):
        sampling = OmniDiffusionSamplingParams(
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=2,
            guidance_scale=1.0,
        )
        if latents is not None:
            sampling.latents = latents[idx]
        requests.append(
            OmniDiffusionRequest(
                request_id=f"req-{idx}",
                prompt={"prompt": prompt},
                sampling_params=sampling,
            )
        )
    return DiffusionRequestBatch(requests=requests)


def test_sdxl_declares_request_batch_support() -> None:
    """SDXL must opt in, otherwise DiffusionEngine rejects max_num_seqs > 1."""
    assert StableDiffusionXLPipeline.supports_request_batch is True


def test_forward_returns_one_output_per_request() -> None:
    recorder: dict = {}
    pipeline = _make_pipeline(recorder)
    batch = _make_batch(["a cat", "a dog", "a bird"])

    outputs = pipeline.forward(batch)

    assert isinstance(outputs, list)
    assert len(outputs) == batch.num_reqs
    assert all(isinstance(out, DiffusionOutput) for out in outputs)
    # All three prompts must ride in a single UNet forward.
    assert recorder["diffuse_batch"] == 3
    assert recorder["encoded_prompts"][0] == ["a cat", "a dog", "a bird"]


def test_forward_splits_output_rows_per_request() -> None:
    recorder: dict = {}
    pipeline = _make_pipeline(recorder)
    batch = _make_batch(["p0", "p1"])

    outputs = pipeline.forward(batch)

    # Request i must receive exactly its own image row (value i), not the
    # whole batched tensor.
    for idx, out in enumerate(outputs):
        assert out.output.shape[0] == 1
        assert torch.allclose(out.output[0], torch.full((3, HEIGHT, WIDTH), float(idx)))


def test_forward_collates_per_request_latents() -> None:
    recorder: dict = {}
    pipeline = _make_pipeline(recorder)
    shape = (1, LATENT_CHANNELS, HEIGHT // VAE_SCALE, WIDTH // VAE_SCALE)
    latents = [torch.full(shape, 1.0), torch.full(shape, 2.0)]
    batch = _make_batch(["p0", "p1"], latents=latents)

    pipeline.forward(batch)

    used = recorder["diffuse_latents"]
    assert used.shape[0] == 2
    assert torch.allclose(used[0], torch.full(shape[1:], 1.0))
    assert torch.allclose(used[1], torch.full(shape[1:], 2.0))


def test_forward_single_request_still_returns_list() -> None:
    recorder: dict = {}
    pipeline = _make_pipeline(recorder)

    outputs = pipeline.forward(_make_batch(["only"]))

    assert len(outputs) == 1
    assert recorder["diffuse_batch"] == 1
