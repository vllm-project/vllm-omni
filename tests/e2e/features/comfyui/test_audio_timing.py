# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from comfyui_vllm_omni.utils.audio_timing import fit_audio_to_length

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("channels", [1, 2, 6])
def test_loop_preserves_each_channel_and_batch(channels):
    waveform = torch.arange(2 * channels * 3, dtype=torch.float64).reshape(2, channels, 3)
    result = fit_audio_to_length(waveform, 8, "loop")
    assert torch.equal(result, waveform[..., torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])])
    assert result.dtype == waveform.dtype and result.device == waveform.device
    assert torch.equal(waveform, torch.arange(2 * channels * 3, dtype=torch.float64).reshape(2, channels, 3))


@pytest.mark.parametrize("policy", ["trim", "loop", "pad"])
def test_all_policies_trim_long_audio(policy):
    waveform = torch.arange(12, dtype=torch.float32).reshape(1, 2, 6)
    assert torch.equal(fit_audio_to_length(waveform, 4, policy), waveform[..., :4])


def test_pad_appends_silence_without_changing_original_samples():
    waveform = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    result = fit_audio_to_length(waveform, 5, "pad")
    assert torch.equal(result[..., :2], waveform)
    assert torch.count_nonzero(result[..., 2:]) == 0
    assert result.shape == (1, 2, 5)


def test_trim_leaves_short_audio_unchanged():
    waveform = torch.tensor([[[1.0, 2.0]]])
    assert torch.equal(fit_audio_to_length(waveform, 5, "trim"), waveform)


@pytest.mark.parametrize("policy", ["trim", "loop", "pad"])
def test_empty_audio_is_rejected(policy):
    with pytest.raises(ValueError, match="at least one sample"):
        fit_audio_to_length(torch.empty(1, 1, 0), 5, policy)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown short-audio policy"):
        fit_audio_to_length(torch.ones(1, 1, 2), 5, "unknown")  # type: ignore[arg-type]
