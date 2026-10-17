# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenEmbedRope,
)
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_txt_seq_lens_from_embeds_uses_padded_width_not_valid_token_count():
    prompt_embeds = torch.zeros(2, 64, 8)
    prompt_embeds_mask = torch.zeros(2, 64, dtype=torch.bool)
    prompt_embeds_mask[:, :10] = True

    assert prompt_embeds_mask.sum(dim=1).tolist() == [10, 10]
    assert txt_seq_lens_from_embeds(prompt_embeds) == [64, 64]


def test_txt_seq_lens_from_embeds_builds_rope_table_for_padded_width():
    padded_width = 32
    prompt_embeds = torch.zeros(2, padded_width, 8)
    txt_seq_lens = txt_seq_lens_from_embeds(prompt_embeds)
    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)

    _, txt_freqs = rope([[(1, 16, 16)]], txt_seq_lens, device="cpu")

    assert txt_freqs.shape[0] == padded_width


def test_txt_seq_lens_from_embeds_supports_2d_embeds():
    prompt_embeds = torch.zeros(48, 16)

    assert txt_seq_lens_from_embeds(prompt_embeds) == [48]


def test_txt_seq_lens_from_embeds_returns_none_for_missing_embeds():
    assert txt_seq_lens_from_embeds(None) is None


def test_txt_seq_lens_from_embeds_rejects_invalid_rank():
    with pytest.raises(ValueError, match="prompt_embeds must be 2D or 3D"):
        txt_seq_lens_from_embeds(torch.zeros(2, 3, 4, 5))


def test_rope_freqs_cache_does_not_outlive_module():
    import gc
    import weakref

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    rope([[(1, 16, 16)]], [8], device="cpu")
    ref = weakref.ref(rope)

    del rope
    gc.collect()

    assert ref() is None


def test_rope_freqs_cache_is_bounded():
    from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import _ROPE_FREQS_CACHE_SIZE

    rope = QwenEmbedRope(theta=10000, axes_dim=[16, 56, 56], scale_rope=True)
    first = rope._compute_video_freqs(1, 2, 2)
    assert rope._compute_video_freqs(1, 2, 2) is first

    for size in range(3, 3 + _ROPE_FREQS_CACHE_SIZE):
        rope._compute_video_freqs(1, size, size)

    assert len(rope._freqs_cache) == _ROPE_FREQS_CACHE_SIZE
    assert (1, 2, 2, 0) not in rope._freqs_cache
