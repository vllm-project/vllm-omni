# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn as nn

from vllm_omni.diffusion.models.z_image import pipeline_z_image
from vllm_omni.diffusion.models.z_image.z_image_transformer import (
    LEARNED_PADDING,
    ZImageTransformer2DModel,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _IdentityRope(nn.Module):
    def forward(self, ids: torch.Tensor):
        values = ids[:, :1].to(torch.float32)
        return values, -values


class _CountingBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, x, *_args):
        self.calls += 1
        return x


def _make_caption_only_transformer() -> ZImageTransformer2DModel:
    transformer = object.__new__(ZImageTransformer2DModel)
    nn.Module.__init__(transformer)
    transformer.cap_embedder = nn.Identity()
    transformer.cap_pad_token = nn.Parameter(torch.full((1, 4), 7.0), requires_grad=False)
    transformer.rope_embedder = _IdentityRope()
    transformer.context_refiner = nn.ModuleList([_CountingBlock()])
    transformer.alignment_padding_mode = LEARNED_PADDING
    return transformer


def test_prepare_caption_context_refines_once_and_preserves_padding():
    transformer = _make_caption_only_transformer()
    caption = torch.arange(12, dtype=torch.float32).reshape(3, 4)

    cap_feats, cap_cos, cap_sin, seq_lens, attn_mask = transformer.prepare_caption_context(
        [caption],
        torch.device("cpu"),
    )

    assert transformer.context_refiner[0].calls == 1
    assert seq_lens == [32]
    assert cap_feats.shape == (1, 32, 4)
    assert torch.equal(cap_feats[0, :3], caption)
    assert torch.equal(cap_feats[0, 3:], torch.full((29, 4), 7.0))
    assert cap_cos.shape == cap_sin.shape == (1, 32, 1)
    assert torch.equal(attn_mask, torch.ones((1, 32), dtype=torch.bool))


def test_cached_caption_context_skips_per_step_caption_preparation(monkeypatch):
    transformer = object.__new__(ZImageTransformer2DModel)
    nn.Module.__init__(transformer)

    def unexpected_prepare(*_args, **_kwargs):
        raise AssertionError("caption inputs were rebuilt for a cached context")

    monkeypatch.setattr(transformer, "_prepare_caption_inputs", unexpected_prepare)
    outputs = transformer.patchify_and_embed(
        [torch.zeros((1, 1, 1, 1))],
        [torch.zeros((3, 4))],
        patch_size=1,
        f_patch_size=1,
        cap_padded_lens=[32],
    )

    assert outputs[1] == []
    assert outputs[4] == []
    assert outputs[6] == []
    assert outputs[7] == []
    assert outputs[3][0][0, 0].item() == 33


def test_request_cache_reuses_cfg_variants_and_skips_subclasses(monkeypatch):
    class FakeTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def prepare_caption_context(self, prompt_embeds, device):
            self.calls += 1
            return (self.calls, prompt_embeds, device)

    class FakeSubclass(FakeTransformer):
        pass

    monkeypatch.setattr(pipeline_z_image, "ZImageTransformer2DModel", FakeTransformer)
    transformer = FakeTransformer()
    cache: dict[bool, tuple] = {}
    prompt_embeds = [torch.zeros((3, 4))]
    device = torch.device("cpu")

    first = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, False, prompt_embeds, device)
    second = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, False, [torch.ones((3, 4))], device)
    cfg = pipeline_z_image._get_or_prepare_caption_context(transformer, cache, True, prompt_embeds, device)

    assert first is second
    assert cfg is not first
    assert transformer.calls == 2

    subclass = FakeSubclass()
    assert pipeline_z_image._get_or_prepare_caption_context(subclass, {}, False, prompt_embeds, device) is None
    assert subclass.calls == 0


def test_cfg_latent_path_is_shared_only_for_equal_padded_caption_lengths(monkeypatch):
    class FakeTransformer(nn.Module):
        pass

    class FakeSubclass(FakeTransformer):
        pass

    monkeypatch.setattr(pipeline_z_image, "ZImageTransformer2DModel", FakeTransformer)
    transformer = FakeTransformer()

    assert pipeline_z_image._can_share_cfg_latent_path(transformer, (None, None, None, [32, 32]), 1)
    assert not pipeline_z_image._can_share_cfg_latent_path(transformer, (None, None, None, [32, 64]), 1)
    assert not pipeline_z_image._can_share_cfg_latent_path(transformer, (None, None, None, [32, 32]), 2)
    assert not pipeline_z_image._can_share_cfg_latent_path(FakeSubclass(), (None, None, None, [32, 32]), 1)


@pytest.mark.parametrize("updated_key", ["prompt_embeds", "negative_prompt_embeds"])
def test_callback_conditioning_update_invalidates_caption_context(updated_key: str):
    latents = torch.zeros(1)
    prompt_embeds = [torch.zeros((2, 4))]
    negative_prompt_embeds = [torch.ones((2, 4))]
    replacement = [torch.full((2, 4), 2.0)]
    caption_context_cache = {False: ("positive",), True: ("cfg",)}

    outputs = {updated_key: replacement}
    _, actual_prompt, actual_negative = pipeline_z_image._apply_step_callback_outputs(
        outputs,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        caption_context_cache,
    )

    assert actual_prompt is (replacement if updated_key == "prompt_embeds" else prompt_embeds)
    assert actual_negative is (replacement if updated_key == "negative_prompt_embeds" else negative_prompt_embeds)
    assert caption_context_cache == {}


def test_latent_only_callback_keeps_caption_context_cache():
    latents = torch.zeros(1)
    replacement = torch.ones(1)
    prompt_embeds = [torch.zeros((2, 4))]
    negative_prompt_embeds = [torch.ones((2, 4))]
    caption_context_cache = {False: ("positive",)}

    actual_latents, _, _ = pipeline_z_image._apply_step_callback_outputs(
        {"latents": replacement},
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        caption_context_cache,
    )

    assert actual_latents is replacement
    assert caption_context_cache == {False: ("positive",)}
