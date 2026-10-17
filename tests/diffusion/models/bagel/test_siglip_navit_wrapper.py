# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The single-stage BAGEL ViT path must reproduce the HF SigLIP vision transformer.

``SiglipNaViTWrapper`` runs the HF encoder on packed patches and has to apply the
``post_layernorm`` the HF model applies before the connector, and the checkpoint's
linear patch embedding (reference ``convert_conv2d_to_linear``: ``(out, p, p, c)``
ordering) has to be converted into the Conv2d layout the wrapper flattens back.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from transformers import SiglipVisionConfig, SiglipVisionModel

from vllm_omni.diffusion.models.bagel.bagel_transformer import patchify
from vllm_omni.diffusion.models.bagel.pipeline_bagel import (
    SiglipNaViTWrapper,
    linear_patch_embedding_to_conv,
)

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]

PATCH = 2
IMAGE = 8
HIDDEN = 32


def _tiny_siglip(seed: int = 0) -> SiglipVisionModel:
    torch.manual_seed(seed)
    config = SiglipVisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        image_size=IMAGE,
        patch_size=PATCH,
        num_channels=3,
        vision_use_head=False,
    )
    config._attn_implementation = "eager"
    model = SiglipVisionModel(config).eval()
    transformer = getattr(model, "vision_model", model)
    with torch.no_grad():
        transformer.post_layernorm.weight.uniform_(0.5, 2.0)
        transformer.post_layernorm.bias.uniform_(-1.0, 1.0)
    return model


def _reference_patchify(image: torch.Tensor, p: int) -> torch.Tensor:
    c, h, w = image.shape
    return torch.einsum("chpwq->hwpqc", image.reshape(c, h // p, p, w // p, p)).reshape(-1, p * p * c)


@torch.no_grad()
def test_wrapper_matches_hf_vision_transformer():
    model = _tiny_siglip()
    image = torch.randn(3, IMAGE, IMAGE)
    expected = model(pixel_values=image[None]).last_hidden_state[0]

    packed = patchify(image, PATCH)
    n = packed.shape[0]
    got = SiglipNaViTWrapper(model)(packed, torch.arange(n), torch.tensor([0, n], dtype=torch.int32), n)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
    pre_norm = model(pixel_values=image[None], output_hidden_states=True).hidden_states[-1][0]
    assert not torch.allclose(got, pre_norm, rtol=1e-3, atol=1e-3)


@torch.no_grad()
def test_wrapper_keeps_packed_images_separate():
    model = _tiny_siglip()
    images = [torch.randn(3, IMAGE, IMAGE) for _ in range(2)]
    expected = torch.cat([model(pixel_values=img[None]).last_hidden_state[0] for img in images])

    packed = torch.cat([patchify(img, PATCH) for img in images])
    n = packed.shape[0] // 2
    ids = torch.cat([torch.arange(n), torch.arange(n)])
    got = SiglipNaViTWrapper(model)(packed, ids, torch.tensor([0, n, 2 * n], dtype=torch.int32), n)

    torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)


def test_linear_patch_embedding_to_conv_inverts_reference_conversion():
    conv = torch.randn(HIDDEN, 3, PATCH, PATCH)
    linear = conv.permute(0, 2, 3, 1).reshape(HIDDEN, 3 * PATCH * PATCH)

    torch.testing.assert_close(linear_patch_embedding_to_conv(linear, tuple(conv.shape)), conv)


def test_converted_patch_embedding_matches_reference_on_packed_patches():
    linear = torch.randn(HIDDEN, 3 * PATCH * PATCH)
    bias = torch.randn(HIDDEN)
    image = torch.randn(3, IMAGE, IMAGE)
    expected = F.linear(_reference_patchify(image, PATCH), linear, bias)

    conv = linear_patch_embedding_to_conv(linear, (HIDDEN, 3, PATCH, PATCH))
    got = F.linear(patchify(image, PATCH), conv.view(HIDDEN, -1), bias)

    torch.testing.assert_close(got, expected)
