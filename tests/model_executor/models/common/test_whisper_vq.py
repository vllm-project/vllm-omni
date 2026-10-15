# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Block causality and padded-frame pooling used by Kimi's audio tokenizer."""

import pytest
import torch
from transformers import WhisperConfig

from vllm_omni.model_executor.models.common.whisper_vq import WhisperVQEncoder


@pytest.fixture
def encoder():
    config = WhisperConfig(
        d_model=8,
        num_mel_bins=4,
        encoder_layers=2,
        encoder_attention_heads=2,
        encoder_ffn_dim=16,
        max_source_positions=16,
        dropout=0,
        attention_dropout=0,
        activation_dropout=0,
        encoder_layerdrop=0,
        encoder_causal_convolution=True,
        pooling_kernel_size=2,
        pooling_type="avg",
        pooling_position=1,
    )
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        return WhisperVQEncoder(config, causal_block_size=4, preserve_padding=True).eval()


@torch.inference_mode()
def test_future_block_does_not_change_prefix(encoder):
    features = torch.linspace(-1, 1, 64).reshape(1, 4, 16)
    changed = features.clone()
    changed[:, :, 8:] += 10
    before = encoder(features).last_hidden_state
    after = encoder(changed).last_hidden_state

    assert before.shape == after.shape == (1, 4, 8)
    torch.testing.assert_close(before[:, :2], after[:, :2], rtol=0, atol=0)
    assert not torch.equal(before[:, 2:], after[:, 2:])


@pytest.mark.parametrize(
    ("preserve_padding", "expected", "expected_mask"),
    [(True, [3.0, 7.0, 5.0], [True, True, False]), (False, [3.0, 3.0], [True, True])],
)
def test_pooling_at_partial_valid_frame_boundary(encoder, preserve_padding, expected, expected_mask):
    encoder._preserve_padding = preserve_padding
    hidden = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0]).reshape(1, 5, 1)
    valid = torch.tensor([[True, True, True, False, False]])

    pooled, mask = encoder._apply_pooling(hidden, valid)

    torch.testing.assert_close(pooled, torch.tensor(expected).reshape(1, -1, 1))
    assert mask.tolist() == [expected_mask]
