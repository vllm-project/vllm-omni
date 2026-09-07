import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import QwenEmbedRope
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
