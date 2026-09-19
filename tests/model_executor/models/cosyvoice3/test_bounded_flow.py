# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import torch
from torch import nn

from vllm_omni.model_executor.models.cosyvoice3.cosyvoice3_code2wav import CosyVoice3Code2Wav
from vllm_omni.transformers_utils.configs.cosyvoice3 import CosyVoice3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class RecordingFlow(nn.Module):
    token_mel_ratio = 2

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward_mel(self, **kwargs):
        self.calls.append(kwargs)
        token = kwargs["token"]
        offset = kwargs["token_offset_tokens"]
        size = token.shape[1] - (0 if kwargs["finalize"] else 3)
        return token[:, offset:size].float().repeat_interleave(2, dim=1).unsqueeze(1)


def make_model(window):
    model = object.__new__(CosyVoice3Code2Wav)
    nn.Module.__init__(model)
    model._flow_context_tokens = window
    model.flow_model = RecordingFlow()
    model._forward_mel = model.flow_model.forward_mel
    model._stream_hift_from_feat = lambda feat, cache_state, finalize: (feat, None if finalize else {})
    return model


def test_bounded_flow_uses_matching_prior_tokens_and_mel():
    model = make_model(4)
    state = None
    prompt_token = torch.arange(10).reshape(1, -1)
    prompt_feat = torch.arange(20).reshape(1, 20, 1).float()
    for offset, total, final in [(0, 9, False), (6, 15, False), (12, 18, True)]:
        waveform, state = model.forward_streaming(
            torch.arange(total).reshape(1, -1),
            prompt_token,
            prompt_feat,
            torch.zeros(1, 2),
            cache_state=state,
            token_offset_tokens=offset,
            finalize=final,
        )
        call = model.flow_model.calls[-1]
        assert call["token"].shape[1] <= 9
        assert call["prompt_token"].shape[1] == 4
        assert call["prompt_feat"].shape[1] == 8
        assert call["token_offset_tokens"] == 0
        if offset:
            torch.testing.assert_close(call["prompt_token"], torch.arange(offset - 4, offset).reshape(1, -1))
            expected = torch.arange(offset - 4, offset).repeat_interleave(2).reshape(1, 8, 1).float()
            torch.testing.assert_close(call["prompt_feat"], expected)
        emitted = total if final else total - 3
        expected = torch.arange(offset, emitted).repeat_interleave(2).reshape(1, 1, -1).float()
        torch.testing.assert_close(waveform, expected)
        if not final:
            assert state["flow_mel"].shape[-1] == 8
    assert state is None


def test_default_keeps_cumulative_flow():
    model = make_model(0)
    model.forward_streaming(
        torch.arange(40).reshape(1, -1),
        torch.zeros(1, 0),
        torch.zeros(1, 0, 1),
        torch.zeros(1, 2),
        token_offset_tokens=24,
    )
    call = model.flow_model.calls[0]
    assert call["token"].shape[1] == 40
    assert call["token_offset_tokens"] == 24


def test_bounded_flow_requires_previous_context():
    model = make_model(4)
    with pytest.raises(ValueError, match="flow_mel"):
        model.forward_streaming(
            torch.ones(1, 8),
            torch.zeros(1, 0),
            torch.zeros(1, 0, 1),
            torch.zeros(1, 2),
            token_offset_tokens=4,
        )


def test_flow_window_config_is_opt_in():
    assert CosyVoice3Config().flow_context_tokens == 0
    assert CosyVoice3Config(flow_context_tokens=24).flow_context_tokens == 24
    with pytest.raises(ValueError, match="nonnegative"):
        CosyVoice3Config(flow_context_tokens=-1)


def test_bounded_batch_keeps_request_contexts_separate():
    model = make_model(4)
    items = []
    for base in (0, 100):
        tokens = torch.arange(base, base + 15).reshape(1, -1)
        prior = tokens[:, 2:6].float().repeat_interleave(2, dim=1).unsqueeze(1)
        items.append(
            {
                "token": tokens,
                "prompt_token": torch.zeros(1, 0),
                "prompt_feat": torch.zeros(1, 0, 1),
                "embedding": torch.zeros(1, 2),
                "cache_state": {"flow_mel": prior},
                "token_offset_tokens": 6,
            }
        )
    results = model.forward_streaming_batch(items)
    for item, (waveform, state), call in zip(items, results, model.flow_model.calls):
        torch.testing.assert_close(call["prompt_token"], item["token"][:, 2:6])
        expected = item["token"][:, 6:12].float().repeat_interleave(2, dim=1).unsqueeze(1)
        torch.testing.assert_close(waveform, expected)
        torch.testing.assert_close(state["flow_mel"], expected[..., -8:])


@pytest.mark.parametrize("prompt_size", [0, 2])
def test_first_chunk_accepts_short_voice_prompt(prompt_size):
    model = make_model(4)
    model.forward_streaming(
        torch.arange(9).reshape(1, -1),
        torch.arange(prompt_size).reshape(1, -1),
        torch.zeros(1, prompt_size * 2, 1),
        torch.zeros(1, 2),
    )
    call = model.flow_model.calls[0]
    assert call["prompt_token"].shape[1] == prompt_size
    assert call["prompt_feat"].shape[1] == prompt_size * 2
