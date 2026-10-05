# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Codec batching must preserve per-request order and convolution boundaries."""

import pytest
import torch

from vllm_omni.model_executor.models.higgs_audio_v3.higgs_audio_v3_code2wav import HiggsAudioV3Code2Wav
from vllm_omni.transformers_utils.configs.higgs_audio_v3 import HiggsAudioV3Config

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture
def codec(monkeypatch):
    model = HiggsAudioV3Code2Wav(config=HiggsAudioV3Config())
    calls = []

    def decode(codes):
        calls.append(tuple(codes.shape))
        if (codes < 0).any():
            raise ValueError("invalid codec row")
        # A time-dependent decoder makes both trim boundaries observable.
        pcm = codes[:, :1].float().repeat_interleave(model.hop_length, dim=-1)
        return pcm

    monkeypatch.setattr(model, "decode_codes", decode)
    return model, calls


def test_ragged_codec_batch_preserves_request_order_and_trims(codec):
    model, calls = codec
    codes = [
        torch.arange(6).expand(8, -1),
        torch.arange(10, 14).expand(8, -1),
        torch.arange(20, 26).expand(8, -1),
    ]
    output = model(
        input_ids=torch.cat([row.reshape(-1) for row in codes]),
        seq_token_counts=[row.numel() for row in codes],
        runtime_additional_information=[
            {"meta": {"left_context_size": 1, "right_holdback_size": 1}},
            {"meta": {"left_context_size": 0, "right_holdback_size": 1}},
            {"meta": {"left_context_size": 1, "right_holdback_size": 1}},
        ],
    )
    assert calls == [(2, 8, 6), (1, 8, 4)]
    for actual, values in zip(
        output.multimodal_outputs["model_outputs"], ([1, 2, 3, 4], [10, 11, 12], [21, 22, 23, 24])
    ):
        expected = torch.tensor(values, dtype=torch.float32).repeat_interleave(model.hop_length)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad_length,bad_value", [(0, 0), (7, 1), (24, -1)])
def test_invalid_codec_request_does_not_discard_valid_peer(codec, bad_length, bad_value):
    model, _ = codec
    valid = torch.arange(3).expand(8, -1).reshape(-1)
    bad = torch.full((bad_length,), bad_value, dtype=torch.long)
    output = model(input_ids=torch.cat([bad, valid]), seq_token_counts=[bad.numel(), valid.numel()])
    audio = output.multimodal_outputs["model_outputs"]
    assert len(audio) == 2
    assert audio[0].numel() == 0
    expected = torch.arange(3, dtype=torch.float32).repeat_interleave(model.hop_length)
    torch.testing.assert_close(audio[1], expected, rtol=0, atol=0)


def test_native_payload_overrides_control_slots_and_empty_terminal(codec):
    model, calls = codec
    codes = torch.arange(6).expand(8, -1).reshape(-1)
    output = model(
        input_ids=torch.zeros(2, dtype=torch.long),
        seq_token_counts=[1, 1],
        runtime_additional_information=[
            {"codes": {"audio": codes}, "meta": {"left_context_size": 1, "right_holdback_size": 1}},
            {"codes": {"audio": torch.empty(0, dtype=torch.long)}, "meta": {"finished": True}},
        ],
    )
    assert calls == [(1, 8, 6)]
    wavs = output.multimodal_outputs["model_outputs"]
    torch.testing.assert_close(wavs[0], torch.arange(1, 5).float().repeat_interleave(model.hop_length))
    assert wavs[1].numel() == 0
