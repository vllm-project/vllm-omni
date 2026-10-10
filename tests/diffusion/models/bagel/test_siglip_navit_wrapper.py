# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image
from transformers import SiglipVisionConfig, SiglipVisionModel

from vllm_omni.diffusion.models.bagel.bagel_transformer import patchify
from vllm_omni.diffusion.models.bagel.pipeline_bagel import SiglipNaViTWrapper, bagel_image_size, bagel_vit_transform

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_wrapper_matches_hf_siglip_forward():
    torch.manual_seed(0)
    config = SiglipVisionConfig(
        hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4, image_size=8, patch_size=2
    )
    model = SiglipVisionModel(config).eval()
    pixels = torch.randn(1, 3, 8, 8)
    ref = model(pixel_values=pixels).last_hidden_state[0]

    packed = patchify(pixels, 2)[0]
    pos_ids = torch.arange(packed.shape[0])
    cu_seqlens = torch.tensor([0, packed.shape[0]], dtype=torch.int32)
    out = SiglipNaViTWrapper(model)(packed, pos_ids, cu_seqlens, packed.shape[0])
    torch.testing.assert_close(out, ref, atol=1e-5, rtol=1e-4)


def test_bagel_image_size_keeps_aspect_and_stride():
    assert bagel_image_size(2000, 1500, 980, 224, 14) == (980, 728)
    w, h = bagel_image_size(300, 3000, 980, 224, 14)
    assert max(w, h) <= 980 and w % 14 == 0 and h % 14 == 0


def test_bagel_vit_transform_resizes_and_normalizes():
    img = Image.effect_noise((2000, 1500), 64).convert("RGB")
    out = bagel_vit_transform(img)
    ref = torch.from_numpy(np.array(img.resize((980, 728), Image.BICUBIC))).permute(2, 0, 1)
    assert out.shape == (3, 728, 980)
    torch.testing.assert_close(out, (ref.float() / 255 - 0.5) / 0.5)  # ToTensor + Normalize(0.5, 0.5)


def test_load_weights_restores_linear_patch_layout():
    from vllm_omni.diffusion.models.bagel.pipeline_bagel import BagelPipeline

    pipeline = BagelPipeline.__new__(BagelPipeline)
    torch.nn.Module.__init__(pipeline)
    pipeline.embeddings = torch.nn.Module()
    pipeline.embeddings.patch_embedding = torch.nn.Conv2d(3, 4, kernel_size=2, stride=2, bias=False)
    # The checkpoint Linear consumes patches flattened in (P, Q, C) order.
    weight = torch.arange(48, dtype=torch.float32).reshape(4, 12)
    name = "embeddings.patch_embedding.weight"
    loaded = pipeline.load_weights([(name, weight)])
    assert name in loaded
    image = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
    patches = image.unfold(2, 2, 2).unfold(3, 2, 2).permute(0, 2, 3, 4, 5, 1).reshape(1, 4, 12)
    expected = torch.nn.functional.linear(patches, weight)
    actual = pipeline.embeddings.patch_embedding(image).flatten(2).transpose(1, 2)
    torch.testing.assert_close(actual, expected)
