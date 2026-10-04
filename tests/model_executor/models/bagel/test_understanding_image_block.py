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
multimodal embedding and ``prepare_runner_inputs`` collapses each run of
image tokens in place, for prefill chunks and for the decode steps replayed
from a CUDA graph alike.
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
    model._image_block_state = {}
    model.device = torch.device("cpu")
    return model


def _step(model, requests: dict[str, tuple[list[int], int]]) -> list[int]:
    ids = [token for tokens, _ in requests.values() for token in tokens]
    raw = [first + i for tokens, first in requests.values() for i in range(len(tokens))]
    positions = torch.tensor(raw)
    _, out = model.prepare_runner_inputs(
        input_ids=torch.tensor(ids),
        positions=positions,
        inputs_embeds=None,
        req_ids=list(requests),
        num_computed_tokens=[first for _, first in requests.values()],
        num_scheduled_tokens=[len(tokens) for tokens, _ in requests.values()],
    )
    assert out is positions
    return positions.tolist()


def test_image_block_shares_one_position_and_decode_continues_from_it():
    model = _model()

    assert _step(model, {"a": ([IMAGE] * 4 + [1, 2, 3], 0)}) == [0, 0, 0, 0, 1, 2, 3]
    assert _step(model, {"a": ([5], 7)}) == [4]
    assert _step(model, {"a": ([6], 8)}) == [5]


@pytest.mark.parametrize("block", [3, 7, 1778])
def test_block_length_follows_the_image(block):
    positions = _step(_model(), {"a": ([IMAGE] * block + [1, 2], 0)})

    assert positions == [0] * block + [1, 2]


def test_text_before_the_image_keeps_its_positions():
    assert _step(_model(), {"a": ([1, 2] + [IMAGE] * 4 + [3], 0)}) == [0, 1, 2, 2, 2, 2, 3]


def test_blocks_separated_by_text_take_one_position_each():
    tokens = [IMAGE] * 3 + [1] + [IMAGE] * 5 + [2]

    assert _step(_model(), {"a": (tokens, 0)}) == [0, 0, 0, 1, 2, 2, 2, 2, 2, 3]


def test_block_split_across_prefill_chunks():
    model = _model()

    assert _step(model, {"a": ([IMAGE] * 3, 0)}) == [0, 0, 0]
    assert _step(model, {"a": ([IMAGE, IMAGE, 1, 2], 3)}) == [0, 0, 1, 2]
    assert _step(model, {"a": ([5], 7)}) == [3]


def test_chunk_boundary_right_after_the_block():
    model = _model()

    assert _step(model, {"a": ([IMAGE] * 4, 0)}) == [0, 0, 0, 0]
    assert _step(model, {"a": ([1, 2], 4)}) == [1, 2]


def test_requests_in_one_step_are_independent():
    model = _model()
    _step(model, {"a": ([IMAGE] * 4 + [1], 0)})

    positions = _step(model, {"a": ([5], 5), "b": ([1, 2, 3], 0), "c": ([IMAGE] * 6 + [4], 0)})

    assert positions == [2, 0, 1, 2, 0, 0, 0, 0, 0, 0, 1]
    assert "b" not in model._image_block_state


def test_finished_request_state_is_dropped():
    model = _model()
    _step(model, {"a": ([IMAGE] * 4, 0)})

    model.on_requests_finished(["a"])

    assert model._image_block_state == {}
    assert _step(model, {"a": ([5], 4)}) == [4]


def test_embedding_inputs_use_the_token_id_buffer():
    model = _model()
    buffer = torch.tensor([IMAGE] * 4 + [1, 2])
    positions = torch.arange(6)

    input_ids, out = model.prepare_runner_inputs(
        input_ids=None,
        positions=positions,
        inputs_embeds=torch.zeros(6, 2),
        req_ids=["a"],
        num_computed_tokens=[0],
        num_scheduled_tokens=[6],
        input_ids_buffer=buffer,
    )

    assert input_ids is buffer
    assert out is positions
    assert positions.tolist() == [0, 0, 0, 0, 1, 2]


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
