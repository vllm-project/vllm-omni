# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""JoyImage GPU inference checks, separated from CPU-only model units."""

import pytest
import torch

from tests.diffusion.models.joy_image import test_joy_image as joy_units
from tests.helpers.mark import hardware_marks
from vllm_omni.diffusion.attention.layer import Attention
from vllm_omni.diffusion.attention.selector import get_attn_backend_for_role
from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import AttentionConfig
from vllm_omni.diffusion.models.joy_image.joy_image_edit_transformer import (
    JoyImageAttention,
    JoyImageEditTransformer3DModel,
)

pytestmark = [
    pytest.mark.core_model,
    pytest.mark.diffusion,
    *hardware_marks(res={"cuda": "L4", "rocm": "MI325"}, num_cards=1),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA or ROCm GPU required"),
]


@pytest.fixture(autouse=True)
def _serving_inference_context():
    # Match DiffusionModelRunner: serving forwards never enable autograd.
    # An empty attention config preserves platform-default backend selection.
    config = joy_units._torch_sdpa_diffusion_config()
    config.diffusion_attention_config = AttentionConfig()
    with torch.inference_mode(), set_current_diffusion_config(config):
        yield


@pytest.mark.parametrize("first_mode", ["dummy", "decoded", "latent"])
def test_layerwise_joy_lifecycle_survives_second_forward(first_mode):
    joy_units._run_tiny_joy_layerwise_sequence(first_mode)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
def test_joy_attention_native_sdpa_matches_torch_sdpa_without_mask(dtype):
    torch.manual_seed(123)
    device = torch.device("cuda")
    attention = joy_units._make_joy_attention(dtype=dtype).to(device=device)
    hidden_states = torch.randn(2, 3, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 5, 32, device=device, dtype=dtype)

    actual = attention(hidden_states, encoder_hidden_states)
    expected = joy_units._reference_joy_attention_output(attention, hidden_states, encoder_hidden_states)

    assert isinstance(attention.attn, Attention)
    assert attention.attn.role == "joy_image.joint"
    assert attention.attn.role_category == "self"
    assert attention.attn.qkv_layout == "BSND"
    joy_units._assert_attention_outputs_close(actual, expected, dtype=dtype)


@pytest.mark.parametrize(
    "dtype",
    [
        torch.float32,
        torch.bfloat16,
    ],
)
def test_joy_attention_native_sdpa_matches_torch_sdpa_with_padding_mask(dtype):
    torch.manual_seed(456)
    device = torch.device("cuda")
    attention = joy_units._make_joy_attention(dtype=dtype).to(device=device)
    hidden_states = torch.randn(2, 3, 32, device=device, dtype=dtype)
    encoder_hidden_states = torch.randn(2, 5, 32, device=device, dtype=dtype)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 0, 0],
        ],
        device=device,
        dtype=torch.bool,
    )

    actual = attention(hidden_states, encoder_hidden_states, attention_mask=attention_mask)
    expected = joy_units._reference_joy_attention_output(
        attention,
        hidden_states,
        encoder_hidden_states,
        attention_mask=attention_mask,
    )

    joy_units._assert_attention_outputs_close(actual, expected, dtype=dtype)


def test_transformer_shape_and_masked_forward():
    device = torch.device("cuda")
    transformer = JoyImageEditTransformer3DModel(
        in_channels=4,
        out_channels=4,
        hidden_size=32,
        text_dim=16,
        num_layers=1,
        num_attention_heads=4,
        patch_size=(1, 2, 2),
    ).to(device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 2, 4, 1, 4, 4, device=device, dtype=torch.bfloat16)
    encoder_hidden_states = torch.randn(2, 5, 16, device=device, dtype=torch.bfloat16)
    encoder_hidden_states_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], device=device)

    output = transformer(
        hidden_states=hidden_states,
        timestep=torch.tensor([1.0, 2.0], device=device),
        encoder_hidden_states=encoder_hidden_states,
        encoder_hidden_states_mask=encoder_hidden_states_mask,
        return_dict=False,
    )[0]

    assert output.shape == hidden_states.shape
    assert isinstance(transformer.double_blocks[0].attn.attn, Attention)


@pytest.mark.parametrize("with_padding_mask", [False, True], ids=["unmasked", "padding-mask"])
def test_joy_attention_platform_default_matches_torch_sdpa(with_padding_mask):
    torch.manual_seed(789)
    dtype = torch.bfloat16
    attention = JoyImageAttention(
        dim=64,
        num_attention_heads=4,
        attention_head_dim=16,
        prefix="double_blocks.0.attn",
    ).to(device="cuda", dtype=dtype)
    expected_backend, _ = get_attn_backend_for_role("joy_image.joint", 16, role_category="self")
    assert attention.attn.attn_backend is expected_backend
    if torch.version.hip is not None:
        # Keep the AITER contract covered instead of silently forcing SDPA.
        assert attention.attn.attn_backend.get_name() == "FLASH_ATTN"

    hidden_states = torch.randn(2, 3, 64, device="cuda", dtype=dtype)
    encoder_hidden_states = torch.randn(2, 5, 64, device="cuda", dtype=dtype)
    attention_mask = None
    if with_padding_mask:
        attention_mask = torch.tensor(
            [[1, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1, 0, 0]],
            device="cuda",
            dtype=torch.bool,
        )

    actual = attention(hidden_states, encoder_hidden_states, attention_mask=attention_mask)
    expected = joy_units._reference_joy_attention_output(
        attention, hidden_states, encoder_hidden_states, attention_mask=attention_mask
    )
    assert not torch.is_grad_enabled()
    assert all(not output.requires_grad for output in actual)
    if attention_mask is None:
        joy_units._assert_attention_outputs_close(actual, expected, dtype=dtype)
        return

    # Self-attention FlashAttention unpads queries as well as keys. SDPA
    # computes padded query rows, but JoyImage never consumes those rows:
    # all image tokens and only the unmasked text tokens are meaningful.
    valid_text = attention_mask[:, hidden_states.shape[1] :]
    valid_actual = (actual[0], actual[1][valid_text])
    valid_expected = (expected[0], expected[1][valid_text])
    joy_units._assert_attention_outputs_close(valid_actual, valid_expected, dtype=dtype)

    # Ignoring padded outputs must not hide a dropped/incorrect key mask.
    # Changing padded text must leave every image and valid text output intact.
    changed_text = encoder_hidden_states.clone()
    changed_text[~valid_text] = 10 * torch.randn_like(changed_text[~valid_text])
    changed_actual = attention(hidden_states, changed_text, attention_mask=attention_mask)
    valid_changed = (changed_actual[0], changed_actual[1][valid_text])
    joy_units._assert_attention_outputs_close(valid_changed, valid_actual, dtype=dtype)
