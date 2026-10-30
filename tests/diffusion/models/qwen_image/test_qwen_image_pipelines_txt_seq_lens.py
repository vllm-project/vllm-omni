"""Behavioral regression tests for Qwen-Image Edit ``txt_seq_lens`` wiring.

RoPE text length must follow padded embed width, not valid-token count from
``mask.sum()``. This exercises the Edit pipeline call site: when masks are
shorter than padded embeds, ``forward()`` must pass padded width into
``diffuse()``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.diffusion.models.qwen_image.pipeline_qwen_image_edit import (
    QwenImageEditPipeline,
)
from vllm_omni.diffusion.worker.request_batch import DiffusionRequestBatch

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]

PADDED_WIDTH = 64
VALID_TOKENS = 10
EMBED_DIM = 8


def _make_padded_prompt_pair():
    prompt_embeds = torch.zeros(1, PADDED_WIDTH, EMBED_DIM)
    prompt_embeds_mask = torch.zeros(1, PADDED_WIDTH, dtype=torch.bool)
    prompt_embeds_mask[:, :VALID_TOKENS] = True
    return prompt_embeds, prompt_embeds_mask


def _make_edit_forward_batch(*, negative_prompt: str | None) -> DiffusionRequestBatch:
    preprocessed = torch.zeros(3, 32, 32)
    return DiffusionRequestBatch(
        requests=[
            SimpleNamespace(
                request_id="edit-txt-seq-lens",
                prompt={
                    "prompt": "edit prompt",
                    "negative_prompt": negative_prompt,
                    "additional_information": {
                        "prompt_image": preprocessed,
                        "preprocessed_image": preprocessed,
                        "calculated_height": 32,
                        "calculated_width": 32,
                    },
                },
                sampling_params=SimpleNamespace(
                    height=32,
                    width=32,
                    num_inference_steps=2,
                    sigmas=None,
                    max_sequence_length=1024,
                    generator=None,
                    true_cfg_scale=4.0,
                    guidance_scale_provided=False,
                    guidance_scale=1.0,
                    num_outputs_per_prompt=1,
                    latents=None,
                    output_type="latent",
                ),
            )
        ]
    )


def _make_stubbed_edit_pipeline():
    pipeline = object.__new__(QwenImageEditPipeline)
    nn.Module.__init__(pipeline)
    pipeline.device = torch.device("cpu")

    pos_embeds, pos_mask = _make_padded_prompt_pair()
    neg_embeds, neg_mask = _make_padded_prompt_pair()

    pipeline.check_inputs = lambda *args, **kwargs: None
    pipeline.check_cfg_parallel_validity = lambda *args, **kwargs: None
    pipeline.vae_scale_factor = 8

    class _GuidanceFreeTransformer:
        guidance_embeds = False
        in_channels = 16

    pipeline.transformer = _GuidanceFreeTransformer()
    pipeline._attention_kwargs = None

    def _fake_encode_prompt(*, prompt_name="prompt", **kwargs):
        if prompt_name == "negative_prompt":
            return neg_embeds, neg_mask
        return pos_embeds, pos_mask

    pipeline.encode_prompt = _fake_encode_prompt

    latents = torch.zeros(1, 4, 8, 8)
    image_latents = torch.zeros(1, 4, 8, 8)
    pipeline.prepare_latents = lambda *args, **kwargs: (latents, image_latents)
    pipeline.prepare_timesteps = lambda *args, **kwargs: (torch.tensor([1.0, 0.5]), 2)

    captured: dict[str, object] = {}

    def _fake_diffuse(
        _prompt_embeds,
        _prompt_embeds_mask,
        _negative_prompt_embeds,
        _negative_prompt_embeds_mask,
        latents_arg,
        _img_shapes,
        txt_seq_lens,
        negative_txt_seq_lens,
        _timesteps,
        do_true_cfg,
        _guidance,
        _true_cfg_scale,
        **kwargs,
    ):
        captured["txt_seq_lens"] = txt_seq_lens
        captured["negative_txt_seq_lens"] = negative_txt_seq_lens
        captured["do_true_cfg"] = do_true_cfg
        return latents_arg

    pipeline.diffuse = _fake_diffuse
    return pipeline, captured, pos_mask, neg_mask


def test_qwen_image_edit_forward_passes_padded_txt_seq_lens_to_diffuse():
    pipeline, captured, pos_mask, neg_mask = _make_stubbed_edit_pipeline()
    batch = _make_edit_forward_batch(negative_prompt="bad prompt")

    pipeline.forward(batch)

    assert pos_mask.sum(dim=1).tolist() == [VALID_TOKENS]
    assert neg_mask.sum(dim=1).tolist() == [VALID_TOKENS]
    assert captured["txt_seq_lens"] == [PADDED_WIDTH]
    assert captured["negative_txt_seq_lens"] == [PADDED_WIDTH]
    assert captured["do_true_cfg"] is True


def test_qwen_image_edit_forward_skips_negative_txt_seq_lens_without_cfg():
    pipeline, captured, pos_mask, _neg_mask = _make_stubbed_edit_pipeline()
    batch = _make_edit_forward_batch(negative_prompt=None)

    pipeline.forward(batch)

    assert pos_mask.sum(dim=1).tolist() == [VALID_TOKENS]
    assert captured["txt_seq_lens"] == [PADDED_WIDTH]
    assert captured["negative_txt_seq_lens"] is None
    assert captured["do_true_cfg"] is False
