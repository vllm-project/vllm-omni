# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import PIL.Image
import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.internvlu.pipeline_internvlu import (
    InternVLUPipeline,
    _preprocess_reference_images,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _LatentDistribution:
    def __init__(self, latents: torch.Tensor) -> None:
        self._latents = latents

    def mode(self) -> torch.Tensor:
        return self._latents


class _FakeVAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            latents_mean=[0.0] * 16,
            latents_std=[1.0] * 16,
        )
        self.encode_shapes: list[tuple[int, ...]] = []
        self.decode_shapes: list[tuple[int, ...]] = []

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def encode(self, pixels: torch.Tensor):
        self.encode_shapes.append(tuple(pixels.shape))
        batch_size, _, _, height, width = pixels.shape
        latents = (
            pixels[:, :1, :1, :1, :1]
            .expand(
                batch_size,
                16,
                1,
                height // 8,
                width // 8,
            )
            .clone()
        )
        return SimpleNamespace(latent_dist=_LatentDistribution(latents))

    def decode(self, latents: torch.Tensor, *, return_dict: bool):
        assert return_dict is False
        self.decode_shapes.append(tuple(latents.shape))
        batch_size = latents.shape[0]
        decoded = latents[:, :1, :1, :1, :1].expand(batch_size, 3, 1, 1, 1).clone()
        return (decoded,)


class _FakeTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.reference_latents: list[list[torch.Tensor]] | None = None

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def prepare_forward_input(
        self,
        hidden_states: list[torch.Tensor],
        image_masks: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        max_length = max(states.shape[0] for states in hidden_states)
        prompt_embeds = torch.zeros(3, max_length, 4096)
        attention_mask = torch.zeros(3, max_length, dtype=torch.bool)
        image_mask = torch.zeros_like(attention_mask)
        for index, (states, mask) in enumerate(zip(hidden_states, image_masks, strict=True)):
            length = states.shape[0]
            prompt_embeds[index, :length] = states
            attention_mask[index, :length] = True
            image_mask[index, :length] = mask
        return prompt_embeds, attention_mask, image_mask

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        self.reference_latents = kwargs["reference_latents"]
        return (torch.zeros_like(hidden_states),)


class _FakeScheduler:
    def __init__(self) -> None:
        self.config = SimpleNamespace(flow_shift=3.0)
        self.timesteps = torch.tensor([500.0])

    def set_timesteps(self, num_steps: int, *, device: torch.device) -> None:
        assert num_steps == 1
        self.timesteps = self.timesteps.to(device)

    def step(
        self,
        velocity: torch.Tensor,
        timestep: torch.Tensor,
        latents: torch.Tensor,
        *,
        return_dict: bool,
    ):
        del velocity, timestep
        assert return_dict is False
        return (latents,)


def _branch(reference_count: int) -> dict[str, torch.Tensor]:
    token_count = max(1, reference_count)
    return {
        "encoder_hidden_states": torch.zeros(token_count, 4096),
        "encoder_image_token_mask": torch.tensor([True] * reference_count + [False] * (token_count - reference_count)),
        "image_fhw_cond": torch.tensor(
            [[1, 1, 1]] * reference_count,
            dtype=torch.long,
        ).reshape(reference_count, 3),
        "reference_image_indices": torch.arange(reference_count),
    }


def _make_edit_batch(images: list[PIL.Image.Image], *, height: int = 128, width: int = 128):
    reference_count = len(images)
    prompt = {
        "prompt": "",
        "height": height,
        "width": width,
        "multi_modal_data": {"image": images},
        "extra": {
            "internvlu": {
                "branches": {
                    "conditional": _branch(reference_count),
                    "partial": _branch(reference_count),
                    "unconditional": _branch(0),
                },
                "image_grid_thw_gen": torch.tensor([[1, height // 8, width // 8]]),
                "requested_height": height,
                "requested_width": width,
                "is_cot": False,
            }
        },
    }
    sampling = SimpleNamespace(
        height=height,
        width=width,
        num_outputs_per_prompt=1,
        generator=None,
        seed=0,
        extra_args={},
        num_inference_steps=1,
    )
    request = SimpleNamespace(
        prompt=prompt,
        sampling_params=sampling,
        is_dummy_run=lambda: False,
    )
    return DiffusionRequestBatch(requests=[request])


def _make_pipeline() -> InternVLUPipeline:
    pipeline = InternVLUPipeline.__new__(InternVLUPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")
    pipeline.vlm_cond_hidden_size = 4096
    pipeline.vae = _FakeVAE()
    pipeline.vae_scale_factor = 8
    pipeline.latent_channels = 16
    pipeline.transformer = _FakeTransformer()
    pipeline.scheduler = _FakeScheduler()
    pipeline.generation_config = {
        "all_cfg_scale": 4.5,
        "part_cfg_scale": 2.0,
        "flow_shift": 3.0,
    }
    return pipeline


def test_reference_preprocessing_preserves_independent_generation_grids():
    landscape = PIL.Image.new("RGB", (1024, 512), color=(255, 0, 0))
    portrait = PIL.Image.new("RGB", (512, 1024), color=(0, 255, 0))

    pixels, generation_grids = _preprocess_reference_images([landscape, portrait])

    assert pixels.shape == (2, 3, 1024, 1024)
    assert torch.equal(
        generation_grids,
        torch.tensor([[1, 64, 128], [1, 128, 64]]),
    )
    # Padding is only a batching concern.  It is zero-valued and lies outside
    # each image's independently resized valid extent.
    assert torch.count_nonzero(pixels[0, :, 512:, :]) == 0
    assert torch.count_nonzero(pixels[1, :, :, 512:]) == 0


def test_forward_crops_independent_reference_grids_and_reuses_encodings():
    pipeline = _make_pipeline()
    landscape = PIL.Image.new("RGB", (1024, 512), color=(255, 0, 0))
    portrait = PIL.Image.new("RGB", (512, 1024), color=(0, 255, 0))

    outputs = pipeline.forward(_make_edit_batch([landscape, portrait]))

    assert len(outputs) == 1
    assert pipeline.vae.encode_shapes == [(2, 3, 1, 1024, 1024)]
    reference_latents = pipeline.transformer.reference_latents
    assert reference_latents is not None
    conditional, partial, unconditional = reference_latents
    assert [tuple(latent.shape) for latent in conditional] == [
        (16, 64, 128),
        (16, 128, 64),
    ]
    assert partial[0] is conditional[0]
    assert partial[1] is conditional[1]
    assert unconditional == []
