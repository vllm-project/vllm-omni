"""Regression tests for #4443: padded prompt width vs RoPE table length.

When ``max(txt_seq_lens) < prompt_embeds.shape[1]``, ``QwenEmbedRope`` builds a
text frequency table that is too short for the padded encoder width.
"""

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenEmbedRope
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

# Representative request-batch padding: valid tokens 10/20, shared width 32.
PADDED_WIDTH = 32
VALID_LENS = [10, 20]


def test_pre_pr_txt_seq_lens_is_shorter_than_padded_width():
    mask = torch.zeros(len(VALID_LENS), PADDED_WIDTH, dtype=torch.bool)
    mask[0, : VALID_LENS[0]] = True
    mask[1, : VALID_LENS[1]] = True

    pre_pr_txt_seq_lens = mask.sum(dim=1).tolist()
    assert pre_pr_txt_seq_lens == VALID_LENS
    assert max(pre_pr_txt_seq_lens) < PADDED_WIDTH


def test_post_pr_txt_seq_lens_matches_padded_width():
    prompt_embeds = torch.zeros(len(VALID_LENS), PADDED_WIDTH, 8)
    assert txt_seq_lens_from_embeds(prompt_embeds) == [PADDED_WIDTH, PADDED_WIDTH]


def test_rope_table_length_matches_padded_width_after_fix():
    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    img_shapes = [[(1, 16, 16)]]

    _, freqs_pre_pr = rope(img_shapes, VALID_LENS, device="cpu")
    _, freqs_post_pr = rope(img_shapes, [PADDED_WIDTH, PADDED_WIDTH], device="cpu")

    assert freqs_pre_pr.shape[0] == max(VALID_LENS)
    assert freqs_pre_pr.shape[0] < PADDED_WIDTH
    assert freqs_post_pr.shape[0] == PADDED_WIDTH
