# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Request-alignment tests for MiniCPM-o 4.5's blocking talker batch."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.utils.mm_outputs import to_payload_element

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class _FakeTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.seen_request_ids: list[str | None] = []

    def forward(
        self,
        input_ids=None,
        positions=None,
        inputs_embeds=None,
        additional_information=None,
    ):
        info = additional_information or {}
        self.seen_request_ids.append(info.get("request_id"))
        if not info.get("emit_audio", False):
            return torch.zeros(input_ids.shape[0], 4)
        value = float(info["waveform_value"])
        return None, torch.full((2,), value, dtype=torch.float32)


def _make_talker_wrapper() -> MiniCPMO45OmniForConditionalGeneration:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    model.model_stage = "tts"
    model.config = SimpleNamespace(hidden_size=4)
    model.talker = _FakeTalker()
    return model


def test_talker_batch_preserves_request_order_and_empty_slots() -> None:
    model = _make_talker_wrapper()
    runtime_info = [
        {"request_id": "req-0", "emit_audio": True, "waveform_value": 1},
        {"request_id": "req-1", "emit_audio": False},
        {"request_id": "req-2", "emit_audio": True, "waveform_value": 3},
    ]

    output = model(
        input_ids=torch.tensor([1, 0, 2, 1, 0, 2, 1, 0, 2]),
        positions=torch.arange(9),
        model_intermediate_buffer=runtime_info,
        runtime_additional_information=[{"request_id": "legacy", "emit_audio": True, "waveform_value": 9}],
    )

    assert model.talker.seen_request_ids == ["req-0", "req-1", "req-2"]
    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert len(waveforms) == len(runtime_info)
    assert waveforms[0].tolist() == [1.0, 1.0]
    assert waveforms[1].numel() == 0
    assert waveforms[2].tolist() == [3.0, 3.0]

    routed = [
        to_payload_element(output.multimodal_outputs, idx, idx * 3, (idx + 1) * 3) for idx in range(len(runtime_info))
    ]
    assert routed[0]["model_outputs"].tolist() == [1.0, 1.0]
    assert routed[1]["model_outputs"].numel() == 0
    assert routed[2]["model_outputs"].tolist() == [3.0, 3.0]


def test_talker_single_request_keeps_batch_aligned_output() -> None:
    model = _make_talker_wrapper()

    output = model(
        input_ids=torch.tensor([1, 0, 2]),
        positions=torch.arange(3),
        runtime_additional_information=[{"request_id": "req-0", "emit_audio": True, "waveform_value": 7}],
    )

    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert len(waveforms) == 1
    assert waveforms[0].tolist() == [7.0, 7.0]


def test_talker_cleans_vocoder_state_after_request_error(mocker) -> None:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)
    talker.audio_tokenizer = SimpleNamespace(
        stream_cache={"stale": True},
        hift_cache_dict={"stale": True},
    )
    mocker.patch.object(talker, "generate_speech", side_effect=RuntimeError("vocoder failed"))

    with pytest.raises(RuntimeError, match="vocoder failed"):
        talker(
            input_ids=torch.tensor([1, 0, 2]),
            additional_information={
                "tts_token_ids": torch.tensor([10]),
                "tts_hidden_states": torch.zeros(1, 4),
            },
        )

    assert talker.audio_tokenizer.stream_cache is None
    assert talker.audio_tokenizer.hift_cache_dict == {}


def test_talker_dummy_logits_keep_one_row_per_request() -> None:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    nn.Module.__init__(talker)

    logits = talker.compute_logits(torch.zeros(3, 4))

    assert logits.shape == (3, 2)
