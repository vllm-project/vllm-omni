# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.lfm3_omni.backbone import Lfm3TemporalModel, causal_conv

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("width", [3, 7])
@pytest.mark.parametrize("splits", [[1, 1, 1, 13], [4, 5, 7], [16]])
def test_conv_chunking_matches_full_reference(width, splits):
    rng = torch.Generator().manual_seed(42)
    x = torch.randn(16, 8, generator=rng)
    weight = torch.randn(8, 1, width, generator=rng)
    bias = torch.randn(8, generator=rng)
    expected = F.conv1d(x.T[None], weight, bias, padding=width - 1, groups=8)[0, :, :16].T
    state, parts, start = None, [], 0
    for length in splits:
        result, state = causal_conv(x[start : start + length], weight, bias, state)
        parts.append(result)
        start += length
    torch.testing.assert_close(torch.cat(parts), expected)
    assert state is not None
    assert state.untyped_storage().nbytes() <= 8 * (width - 1) * x.element_size()


def test_conv_request_states_are_independent():
    torch.manual_seed(7)
    x, y = torch.randn(9, 8), torch.randn(11, 8)
    weight = torch.randn(8, 1, 7)
    x0, xs = causal_conv(x[:4], weight, None, None)
    y0, ys = causal_conv(y[:6], weight, None, None)
    x1, _ = causal_conv(x[4:], weight, None, xs)
    y1, _ = causal_conv(y[6:], weight, None, ys)
    torch.testing.assert_close(torch.cat((x0, x1)), causal_conv(x, weight, None, None)[0])
    torch.testing.assert_close(torch.cat((y0, y1)), causal_conv(y, weight, None, None)[0])


def test_rope_zero_tail_preserves_both_unrotated_halves():
    x = torch.arange(128, dtype=torch.float32).reshape(1, 1, 128)
    # Only 32 of 64 inverse frequencies are rotated; both halves have a tail.
    angles = torch.cat((torch.arange(32).float() / 128, torch.zeros(32)))
    angles = torch.cat((angles, angles))[None]
    result = Lfm3TemporalModel._rotate(x, angles.cos(), angles.sin())
    torch.testing.assert_close(result[..., 32:64], x[..., 32:64], rtol=0, atol=0)
    torch.testing.assert_close(result[..., 96:128], x[..., 96:128], rtol=0, atol=0)
    assert not torch.equal(result[..., :32], x[..., :32])


@pytest.mark.parametrize("window", [None, 4])
def test_reference_attention_chunking_and_sliding_cache(window):
    from unittest.mock import Mock

    from vllm_omni.model_executor.models.lfm3_omni.backbone import RequestState

    model = Lfm3TemporalModel.__new__(Lfm3TemporalModel)
    torch.nn.Module.__init__(model)
    model.attention_mode = "reference"
    model.config = Mock(
        num_key_value_heads=1,
        head_dim=4,
        sliding_window=window,
        layer_types=["full_attention" if window is None else "sliding_attention"],
    )

    def attention(module, q, k, v, mask, scaling, dropout):
        output = F.scaled_dot_product_attention(
            q, k.repeat_interleave(2, 1), v.repeat_interleave(2, 1), mask, scale=scaling
        )
        return output.transpose(1, 2), None

    model._reference_attention = attention
    attn = Mock(scaling=0.5)
    generator = torch.Generator().manual_seed(31)
    q = torch.randn(13, 2, 4, generator=generator)
    k = torch.randn(13, 1, 4, generator=generator)
    v = torch.randn(13, 4, generator=generator)
    state = RequestState()
    outputs = []
    start = 0
    for length in [5, 1, 3, 4]:
        end = start + length
        outputs.append(
            model._attention(
                0, attn, q[start:end], k[start:end], v[start:end], torch.arange(start, end), [(0, length, state)]
            )
        )
        if window is not None:
            assert state.kv[0][0].shape[-2] <= window - 1
        start = end
    positions = torch.arange(13)
    allowed = positions[None] <= positions[:, None]
    if window is not None:
        allowed &= positions[None] > positions[:, None] - window
    expected = (
        F.scaled_dot_product_attention(
            q.transpose(0, 1)[None],
            k.transpose(0, 1)[None].repeat_interleave(2, 1),
            v.T.reshape(1, 1, 4, 13).transpose(-1, -2).repeat_interleave(2, 1),
            allowed,
            scale=0.5,
        )[0]
        .transpose(0, 1)
        .flatten(1)
    )
    torch.testing.assert_close(torch.cat(outputs), expected, rtol=1e-5, atol=1e-6)
