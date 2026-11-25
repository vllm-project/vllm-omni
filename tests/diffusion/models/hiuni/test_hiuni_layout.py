# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks for HiUni token layouts, attention masks, sigmas and pixel patch helpers."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.hiuni.pipeline_hiuni import (
    BOI_ID,
    EOI_ID,
    IM_END_ID,
    IM_START_ID,
    IMG_PAD_ID,
    TIME_TOKEN_ID,
    HiUniPipeline,
    build_attention_mask,
    build_positions,
    build_startend,
    chat_visible_from,
    decode_image,
    edit_input_ids,
    expand_image_pads,
    hiuni_sigmas,
    sample_token,
    t2i_input_ids,
    t2i_layout,
)
from vllm_omni.diffusion.utils.hf_utils import _looks_like_hiuni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_sigmas_start_at_one_and_decrease():
    sigmas = hiuni_sigmas(25, 4.0)
    assert sigmas.shape == (25,)
    assert sigmas[0].item() == pytest.approx(1.0)
    assert torch.all(sigmas[1:] < sigmas[:-1])
    plain = hiuni_sigmas(4, 1.0)
    assert torch.allclose(plain, torch.linspace(1.0, 0.003 / 1.002, 5)[:-1])


def test_t2i_ids_negative_branch_is_the_image_block():
    ids = t2i_input_ids(_CharTokenizer(), None, seed=0, height=2, width=3)
    assert ids == [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * 6 + [EOI_ID, IM_END_ID]


def test_t2i_ids_positive_branch_is_seeded_chat():
    first = t2i_input_ids(_CharTokenizer(), "cat", seed=7, height=2, width=2)
    assert first == t2i_input_ids(_CharTokenizer(), "cat", seed=7, height=2, width=2)
    assert first[0] == IM_START_ID
    assert first[-8:] == [BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * 4 + [EOI_ID, IM_END_ID]
    assert first.count(IMG_PAD_ID) == 4


def test_t2i_layout_positions_and_bidirectional_block():
    ids = torch.tensor([1, 2, BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * 6 + [EOI_ID, IM_END_ID])
    positions, mask = t2i_layout(ids, 2, 3, torch.float32, torch.device("cpu"))
    start, end = 4, 10
    assert positions.shape == (3, 1, ids.numel())
    assert torch.all(positions[0, 0, start:end] == start)
    assert positions[1, 0, start:end].tolist() == [4, 4, 4, 5, 5, 5]
    assert positions[2, 0, start:end].tolist() == [4, 5, 6, 4, 5, 6]
    assert positions[0, 0, end:].tolist() == [7, 8]
    visible = mask[0, 0] == 0
    assert visible[start, end - 1]
    assert not visible[0, 1]
    assert visible[end, 0]


def test_edit_ids_and_grids():
    ids, grids = edit_input_ids(_CharTokenizer(), "ab", (2, 2), 1, 3)
    assert grids == [[1, 2, 2], [1, 1, 3]]
    assert ids.count(IMG_PAD_ID) == 4 + 3
    assert ids.count(TIME_TOKEN_ID) == 1
    uncond, uncond_grids = edit_input_ids(_CharTokenizer(), None, None, 1, 3)
    assert uncond_grids == [[1, 1, 3]]
    assert uncond[0] == BOI_ID


def test_noisy_block_is_hidden_after_its_end():
    ids = [7, BOI_ID] + [IMG_PAD_ID] * 2 + [EOI_ID, 8, BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * 2 + [EOI_ID, 9]
    ute, lts = build_startend(ids)
    mask = build_attention_mask(ute, lts, torch.float32, torch.device("cpu"))[0, 0] == 0
    clean, noisy, last = 2, 8, len(ids) - 1
    # Image patches attend bidirectionally inside their own block.
    assert mask[2, 3] and mask[8, 9]
    # The clean reference stays visible to later rows; the noisy block does not.
    assert mask[last, clean]
    assert not mask[last, noisy]
    assert mask[noisy, clean]


def test_build_positions_ramps_image_blocks():
    ids = [5, IMG_PAD_ID, IMG_PAD_ID, IMG_PAD_ID, IMG_PAD_ID, 6]
    positions = build_positions(ids, [[1, 2, 2]])
    assert positions[:, 0].tolist() == [0, 0, 0]
    assert positions[:, 1:5].tolist() == [[1, 1, 1, 1], [1, 1, 2, 2], [1, 2, 1, 2]]
    assert positions[:, 5].tolist() == [3, 3, 3]


def test_chat_image_pads_expand_with_time_token():
    ids = expand_image_pads([1, BOI_ID, IMG_PAD_ID, EOI_ID, 2], [[1, 1, 3]])
    assert ids == [1, BOI_ID, TIME_TOKEN_ID] + [IMG_PAD_ID] * 3 + [EOI_ID, 2]
    ute = chat_visible_from(torch.tensor(ids))
    assert ute.tolist() == [0, 1, 2, 3, 3, 3, 6, 7]


def test_decode_image_shapes_and_resize():
    latents = torch.zeros(4 * 2 * 3, 768)
    image = decode_image(latents, 64, 96)
    assert image.size == (96, 64)
    assert image.getpixel((0, 0)) == (128, 128, 128)
    assert decode_image(latents, 64, 96, size=(50, 40)).size == (50, 40)


def test_auto_edit_size_is_not_an_explicit_size():
    def sizes(**kwargs):
        req = SimpleNamespace(sampling_params=OmniDiffusionSamplingParams(height=576, width=480, **kwargs))
        sampling = HiUniPipeline._sampling(HiUniPipeline, req)
        return sampling["height"], sampling["width"]

    assert sizes() == (576, 480)
    assert sizes(height_not_provided=True, width_not_provided=True) == (None, None)


def test_sample_token_greedy_and_top_k_one():
    logits = torch.tensor([0.1, 3.0, 0.2])
    assert sample_token(logits, 0.0, 0, 1.0, None) == 1
    generator = torch.Generator().manual_seed(0)
    assert sample_token(logits, 1.0, 1, 1.0, generator) == 1


@pytest.mark.parametrize(
    "config, expected",
    [
        ({"model_type": "pixel_dit_qwen3_vl"}, True),
        ({"architectures": ["PixelDiTForConditionalGeneration"]}, True),
        ({"model_type": "qwen3_vl", "architectures": ["Qwen3VLForConditionalGeneration"]}, False),
    ],
)
def test_looks_like_hiuni(config, expected):
    assert _looks_like_hiuni(config) is expected
