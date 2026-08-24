# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for Bagel.forward_cache_update_vae packed-sequence assembly."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import PretrainedConfig
from vllm.transformers_utils.configs.bagel import BagelConfig

from vllm_omni.diffusion.models.bagel.bagel_transformer import (
    Bagel,
    BaseNavitOutputWithPast,
    NaiveCache,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN_SIZE = 32
NUM_LAYERS = 2


def _make_bagel_config() -> BagelConfig:
    llm_config = PretrainedConfig()
    llm_config.hidden_size = HIDDEN_SIZE
    llm_config.layer_module = "Qwen2MoTDecoderLayer"
    llm_config.num_hidden_layers = NUM_LAYERS
    llm_config.num_attention_heads = 4

    vae_config = PretrainedConfig()
    vae_config.downsample = 8
    vae_config.z_channels = 4

    return BagelConfig(
        llm_config=llm_config,
        vae_config=vae_config,
        visual_gen=True,
        visual_und=False,
        latent_patch_size=2,
        max_latent_size=32,
    )


def _make_tracking_language_model() -> MagicMock:
    """Return a mock LM that records forward() kwargs."""
    calls: list[dict] = []
    hidden_size = HIDDEN_SIZE

    def forward(*_args, **kwargs):
        calls.append(dict(kwargs))
        if kwargs.get("return_embeddings_only"):
            text_ids = kwargs["packed_text_ids"]
            emb = torch.randn(text_ids.shape[0], hidden_size)
            return BaseNavitOutputWithPast(packed_query_sequence=emb)

        packed_query_sequence = kwargs.get("packed_query_sequence")
        if packed_query_sequence is None:
            text_ids = kwargs["packed_text_ids"]
            packed_query_sequence = torch.randn(text_ids.shape[0], hidden_size)
        return BaseNavitOutputWithPast(
            packed_query_sequence=packed_query_sequence,
            past_key_values=kwargs.get("past_key_values"),
        )

    lm = MagicMock()
    lm.forward = forward
    lm._calls = calls
    return lm


def _vae_transform(image: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(image.convert("RGB"), dtype=np.float32))
    arr = arr / 127.5 - 1.0
    return arr.permute(2, 0, 1)


@pytest.fixture
def bagel_and_vae_input():
    language_model = _make_tracking_language_model()
    bagel = Bagel(language_model=language_model, vit_model=None, config=_make_bagel_config())

    image = Image.new("RGB", (64, 64), color=(128, 64, 32))
    new_token_ids = {"start_of_image": 100, "end_of_image": 101}

    gen_input, _newlens, _new_rope = bagel.prepare_vae_images(
        curr_kvlens=[0],
        curr_rope=[0],
        images=[image],
        transforms=_vae_transform,
        new_token_ids=new_token_ids,
    )

    vae_model = MagicMock()
    patch_shapes = gen_input["patchified_vae_latent_shapes"]
    p = bagel.latent_patch_size
    c = bagel.latent_channel

    def encode(padded_images: torch.Tensor) -> torch.Tensor:
        # Match AutoEncoder.encode layout: [B, z_channels, h*p, w*p] per image.
        batch = padded_images.shape[0]
        assert batch == len(patch_shapes)
        latents = [torch.randn(c, h * p, w * p) for h, w in patch_shapes]
        return torch.stack(latents, dim=0)

    vae_model.encode = encode

    expected_seq_len = int(gen_input["packed_seqlens"].sum().item())
    num_text_tokens = int(gen_input["packed_text_ids"].numel())

    return SimpleNamespace(
        bagel=bagel,
        language_model=language_model,
        vae_model=vae_model,
        gen_input=gen_input,
        past_key_values=NaiveCache(NUM_LAYERS),
        expected_seq_len=expected_seq_len,
        num_text_tokens=num_text_tokens,
    )


class TestForwardCacheUpdateVae:
    """Regression tests for single-stage img2img VAE cache prefill."""

    def test_forwards_full_packed_query_sequence(self, bagel_and_vae_input):
        """Update forward must use packed_query_sequence spanning text + VAE tokens."""
        ctx = bagel_and_vae_input
        assert ctx.expected_seq_len > ctx.num_text_tokens, "test setup should include many VAE tokens"

        ctx.bagel.forward_cache_update_vae(
            ctx.vae_model,
            ctx.past_key_values,
            **ctx.gen_input,
        )

        update_calls = [c for c in ctx.language_model._calls if not c.get("return_embeddings_only")]
        assert len(update_calls) == 1

        call = update_calls[0]
        assert "packed_query_sequence" in call
        assert "packed_text_ids" not in call

        packed = call["packed_query_sequence"]
        assert packed.shape[0] == ctx.expected_seq_len
        assert call.get("mode") == "gen"

    def test_packed_sequence_covers_text_and_vae_index_ranges(self, bagel_and_vae_input):
        """Text and VAE index ranges must lie within packed_query_sequence."""
        ctx = bagel_and_vae_input
        ctx.bagel.forward_cache_update_vae(
            ctx.vae_model,
            ctx.past_key_values,
            **ctx.gen_input,
        )

        update_call = next(c for c in ctx.language_model._calls if not c.get("return_embeddings_only"))
        packed = update_call["packed_query_sequence"]

        text_idx = ctx.gen_input["packed_text_indexes"]
        vae_idx = ctx.gen_input["packed_vae_token_indexes"]
        assert int(text_idx.max()) < packed.shape[0]
        assert int(vae_idx.max()) < packed.shape[0]
        assert int(text_idx.min()) >= 0
        assert int(vae_idx.min()) >= 0
