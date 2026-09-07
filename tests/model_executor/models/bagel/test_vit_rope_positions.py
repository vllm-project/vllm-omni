# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BAGEL packs an image block -- <|vision_start|>, the ViT patches and <|vision_end|> -- at a single
RoPE position and continues the text at the next one (Bagel/modeling/bagel/bagel.py,
prepare_vit_images); an img2img input is a VAE block, a separator and a ViT block at two positions.
vLLM's default is one position per token; the Thinker provides its own via get_mrope_input_positions."""

import torch
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration


def feature(modality, offset, length, is_embed=None):
    return MultiModalFeatureSpec(
        data=None,
        modality=modality,
        identifier="x",
        mm_position=PlaceholderRange(offset=offset, length=length, is_embed=is_embed),
    )


def positions(num_tokens, features):
    pos, delta = OmniBagelForConditionalGeneration.get_mrope_input_positions(None, [0] * num_tokens, features)
    assert pos.shape == (3, num_tokens) and torch.equal(pos[0], pos[1]) and torch.equal(pos[0], pos[2])
    return pos[0].tolist(), delta


def test_image_block_shares_one_position():
    # T [VS P P P P VE] T T
    pos, delta = positions(9, [feature("image", 1, 6)])
    assert pos == [0, 1, 1, 1, 1, 1, 1, 2, 3], "ViT tokens must not get one position per token"
    assert delta == 4 - 9  # the first decoded token continues at position 4


def test_two_images_and_text_only():
    pos, _ = positions(8, [feature("image", 1, 3), feature("image", 4, 3)])
    assert pos == [0, 1, 1, 1, 2, 2, 2, 3]
    assert positions(4, [])[0] == [0, 1, 2, 3]


def test_interleaved_text_and_images():
    # T T [VS P P VE] T T [VS P P VE] T
    pos, delta = positions(13, [feature("image", 2, 4), feature("image", 8, 4)])
    assert pos == [0, 1, 2, 2, 2, 2, 3, 4, 5, 5, 5, 5, 6]
    assert delta == 7 - 13


def test_img2img_vae_separator_vit_layout():
    # T T [VAE block | separator | ViT block] T : VAE block and separator at 2, ViT block at 3, text from 4
    is_embed = torch.tensor([True] * 3 + [False] + [True] * 3)
    pos, delta = positions(10, [feature("img2img", 2, 7, is_embed)])
    assert pos == [0, 1, 2, 2, 2, 2, 3, 3, 3, 4]
    assert delta == 5 - 10
