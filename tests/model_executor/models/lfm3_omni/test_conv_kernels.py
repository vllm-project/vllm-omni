# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import types

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.lfm3_omni.backbone import causal_conv

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def _checkpoint_kernels(calls):
    """Stands in for the checkpoint's CUDA width-7 kernels with the same shapes and state contract."""

    def prefill(window, taps, bias):
        calls.append(("prefill", window.shape[-1]))
        return F.conv1d(window, taps[:, None], bias, padding=taps.shape[-1] - 1, groups=window.shape[1])[
            ..., : window.shape[-1]
        ]

    def update(x, state, taps, bias):
        calls.append(("decode", state.shape[-1]))
        window = torch.cat((state, x), dim=-1)
        state.copy_(window[..., 1:])
        out = (window[..., -taps.shape[-1] :] * taps[None]).sum(-1, keepdim=True)
        return out if bias is None else out + bias[None, :, None]

    module = types.ModuleType("lfm3_conv_stub")
    module.__dict__.update(
        can_use_lfm3_conv=lambda x, w: True, lfm3_causal_conv1d=prefill, lfm3_causal_conv1d_update=update
    )
    return module


@pytest.mark.parametrize("with_bias", [False, True])
def test_checkpoint_kernels_cover_prompt_continuation_and_decode(with_bias):
    rng = torch.Generator().manual_seed(3)
    x = torch.randn(12, 8, generator=rng)
    weight = torch.randn(8, 1, 7, generator=rng)
    bias = torch.randn(8, generator=rng) if with_bias else None
    expected = F.conv1d(x.T[None], weight, bias, padding=6, groups=8)[0, :, :12].T
    calls: list[tuple[str, int]] = []
    kernels = _checkpoint_kernels(calls)
    prompt, state = causal_conv(x[:5], weight, bias, None, kernels)
    # A multi-token chunk continuing the prompt uses the prompt kernel over the retained inputs.
    continued, state = causal_conv(x[5:8], weight, bias, state, kernels)
    steps = []
    for t in range(8, 12):
        out, next_state = causal_conv(x[t : t + 1], weight, bias, state, kernels)
        assert next_state is state  # updated in place, as the reference cache is
        steps.append(out)
    torch.testing.assert_close(torch.cat([prompt, continued, *steps]), expected)
    # The kernels keep the reference's seven-input cache.
    assert state.shape[-1] == 7
    assert calls == [("prefill", 11), ("prefill", 10)] + [("decode", 7)] * 4
