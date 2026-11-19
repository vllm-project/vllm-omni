# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""The AR stage must lay out an input image like the reference.

``Bagel.prepare_vit_images`` packs ``<|vision_start|>`` + ViT tokens +
``<|vision_end|>`` into one block: the markers attend bidirectionally with the
ViT tokens, the whole block takes a single position id, and the text that
follows continues from the next one. The ViT is a NaViT: an image keeps its
aspect ratio and patch ``(row, col)`` reads position ``row * 70 + col`` of the
70 x 70 table, so blocks differ in length from image to image.

The AR engine numbers tokens sequentially, so the markers are part of the
multimodal embedding and MRv2 stores the compressed positions for the full prompt, then selects the
scheduled prefill slices and applies the same decode delta on every step.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.bagel.bagel import OmniBagelForConditionalGeneration, _image_tensors

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

IMAGE = 9


def _model() -> OmniBagelForConditionalGeneration:
    model = object.__new__(OmniBagelForConditionalGeneration)
    nn.Module.__init__(model)
    model._image_token_id = IMAGE
    model.device = torch.device("cpu")
    return model


@pytest.mark.parametrize(
    "tokens, expected",
    [
        ([IMAGE] * 4 + [1, 2, 3], [0, 0, 0, 0, 1, 2, 3]),
        ([1, 2] + [IMAGE] * 4 + [3], [0, 1, 2, 2, 2, 2, 3]),
        ([IMAGE] * 3 + [1] + [IMAGE] * 5 + [2], [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]),
        ([1, 2, 3], [0, 1, 2]),
        ([], []),
    ],
)
def test_full_prompt_positions_and_decode_delta(tokens, expected):
    positions, delta = _model().get_mrope_input_positions(tokens, [])
    assert positions.tolist() == [expected]
    assert len(tokens) + delta == (expected[-1] + 1 if expected else 0)


@pytest.mark.parametrize("block", [3, 7, 1778])
def test_block_length_follows_the_image(block):
    positions, delta = _model().get_mrope_input_positions([IMAGE] * block + [1, 2], [])
    assert positions.tolist() == [[0] * block + [1, 2]]
    assert delta == 1 - block


def test_recomputed_prompt_does_not_accumulate_position_shift():
    model = _model()
    tokens = [IMAGE] * 4 + [1, 2]
    first, first_delta = model.get_mrope_input_positions(tokens, [])
    model.get_mrope_input_positions([IMAGE] * 8 + [3], [])
    replay, replay_delta = model.get_mrope_input_positions(tokens, [])
    assert torch.equal(first, replay)
    assert first_delta == replay_delta == -3


class _Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.inputs: list[torch.Tensor] = []

    def forward(self, inputs_embeds, return_all_hidden_states):
        assert not return_all_hidden_states
        self.inputs.append(inputs_embeds)
        return inputs_embeds * 2


def test_vit_reads_navit_positions_for_a_non_square_image():
    torch.manual_seed(0)
    side, patch, dim = 5, 2, 4
    model = _model()
    model.config = SimpleNamespace(vit_max_num_patch_per_side=side)
    vision = nn.Module()
    vision.embeddings = nn.Module()
    vision.embeddings.patch_embedding = nn.Conv2d(3, dim, patch, patch)
    vision.embeddings.position_embedding = nn.Embedding(side * side, dim)
    vision.encoder = _Encoder()
    vision.post_layernorm = nn.LayerNorm(dim)
    model.vit_model = SimpleNamespace(vision_model=vision)
    model.connector = nn.Linear(dim, dim)
    table = torch.randn(side * side, dim)
    model.vit_pos_embed = lambda ids: table[ids]
    pixels = torch.randn(3, 3 * patch, 2 * patch)

    with torch.no_grad():
        out = model._encode_vit(pixels)
        ids = torch.tensor([0, 1, 5, 6, 10, 11])
        expected_input = vision.embeddings.patch_embedding(pixels[None]).flatten(2).transpose(1, 2)
        expected_input = expected_input + vision.embeddings.position_embedding(ids)
        expected = model.connector(vision.post_layernorm(expected_input * 2)[0]) + table[ids]

    torch.testing.assert_close(vision.encoder.inputs[0], expected_input)
    torch.testing.assert_close(out, expected)


def test_understanding_image_embedding_carries_the_vision_markers():
    torch.manual_seed(0)
    model = _model()
    embed_tokens = nn.Embedding(8, 4)
    model.language_model = SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens))
    model._start_of_image_id = 3
    model._end_of_image_id = 5
    vit = {6: torch.randn(6, 4), 2: torch.randn(2, 4)}
    model._encode_vit = lambda pixel_values: vit[pixel_values.shape[-1]]

    with torch.no_grad():
        blocks = model._process_img2text_input({"pixel_values": [torch.zeros(3, 4, 6), torch.zeros(3, 4, 2)]})

    assert [tuple(block.shape) for block in blocks] == [(8, 4), (4, 4)]
    for block, image in zip(blocks, vit.values()):
        torch.testing.assert_close(block[0], embed_tokens.weight[3])
        torch.testing.assert_close(block[1:-1], image)
        torch.testing.assert_close(block[-1], embed_tokens.weight[5])


@pytest.mark.parametrize(
    ("pixel_values", "shapes"),
    [
        (torch.zeros(2, 3, 4, 6), [(3, 4, 6), (3, 4, 6)]),
        (torch.zeros(1, 2, 3, 4, 6), [(3, 4, 6), (3, 4, 6)]),
        (torch.zeros(3, 4, 6), [(3, 4, 6)]),
        ([torch.zeros(3, 4, 6), [torch.zeros(3, 8, 2)]], [(3, 4, 6), (3, 8, 2)]),
    ],
    ids=["batched", "nested-batch", "single", "variable-sizes"],
)
def test_image_tensors_flattens_every_layout(pixel_values, shapes):
    assert [tuple(t.shape) for t in _image_tensors(pixel_values)] == shapes
