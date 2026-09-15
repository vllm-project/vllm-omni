import pytest
import torch

from vllm_omni.diffusion.models.qwen_image.qwen_image_transformer import (
    QwenEmbedRope,
    _apply_qwen_image_rotary_emb,
)
from vllm_omni.diffusion.models.qwen_image.rope_utils import txt_seq_lens_from_embeds

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_qwen_rotary_preserves_frequency_precision(dtype, noncontiguous):
    from diffusers.models.transformers.transformer_qwenimage import apply_rotary_emb_qwen

    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, 17, 3, 128, generator=generator).to(dtype)
    if noncontiguous:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    angles = torch.randn(17, 64, generator=generator)
    freqs = torch.polar(torch.ones_like(angles), angles)
    expected = apply_rotary_emb_qwen(x, freqs, use_real=False)
    actual = _apply_qwen_image_rotary_emb(x, freqs)
    assert actual.dtype == dtype
    assert actual.shape == x.shape
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    if dtype == torch.bfloat16:
        # A regression to early BF16 frequency rounding must be observable.
        rounded_freqs = torch.complex(freqs.real.to(dtype).float(), freqs.imag.to(dtype).float())
        rounded = _apply_qwen_image_rotary_emb(x, rounded_freqs)
        assert not torch.equal(actual, rounded)


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
