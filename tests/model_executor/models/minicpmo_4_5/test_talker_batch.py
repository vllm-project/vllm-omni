# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch
import torch.nn as nn

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni import (
    MiniCPMO45OmniForConditionalGeneration,
)
from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    MiniCPMO45OmniTTSForConditionalGeneration,
)
from vllm_omni.model_executor.models.output_templates import OmniOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@dataclass(frozen=True)
class _Config:
    hidden_size: int = 4


class _RecordingTalker(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[dict[str, object]] = []

    def forward(self, *, additional_information: dict[str, object], **_: object) -> tuple[None, torch.Tensor]:
        self.calls.append(additional_information)
        if additional_information.get("raise_error"):
            raise RuntimeError("talker failed")
        marker_value = additional_information["marker"]
        assert isinstance(marker_value, int)
        marker = float(marker_value)
        return None, torch.tensor([marker], dtype=torch.float32)


def _make_model() -> tuple[MiniCPMO45OmniForConditionalGeneration, _RecordingTalker]:
    model = MiniCPMO45OmniForConditionalGeneration.__new__(MiniCPMO45OmniForConditionalGeneration)
    nn.Module.__init__(model)
    talker = _RecordingTalker()
    model.model_stage = "tts"
    model.config = _Config()
    model.talker = talker
    return model, talker


def _info(marker: int, *, raise_error: bool = False) -> dict[str, object]:
    return {
        "marker": marker,
        "raise_error": raise_error,
        "tts_token_ids": torch.tensor([marker]),
        "tts_hidden_states": torch.tensor([[float(marker)]]),
    }


def _forward(
    model: MiniCPMO45OmniForConditionalGeneration,
    infos: list[dict[str, object]],
    **kwargs: object,
) -> OmniOutput:
    return model(
        input_ids=torch.tensor([1, 0, 2]),
        positions=torch.arange(3),
        runtime_additional_information=infos,
        **kwargs,
    )


def test_talker_outputs_follow_request_order() -> None:
    model, talker = _make_model()

    output = _forward(model, [_info(11), _info(22)])

    assert [call["marker"] for call in talker.calls] == [11, 22]
    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert [waveform.tolist() for waveform in waveforms] == [[11.0], [22.0]]


def test_missing_tts_input_keeps_empty_output_slot() -> None:
    model, talker = _make_model()

    output = _forward(model, [_info(11), {}, _info(33)])

    assert [call["marker"] for call in talker.calls] == [11, 33]
    assert output.multimodal_outputs is not None
    waveforms = output.multimodal_outputs["model_outputs"]
    assert len(waveforms) == 3
    assert waveforms[0].tolist() == [11.0]
    assert waveforms[1].dtype == torch.float32
    assert waveforms[1].numel() == 0
    assert waveforms[2].tolist() == [33.0]


def test_talker_exception_propagates() -> None:
    model, _ = _make_model()

    with pytest.raises(RuntimeError, match="talker failed"):
        _forward(model, [_info(11), _info(22, raise_error=True)])


def test_consecutive_invocations_do_not_reuse_request_metadata() -> None:
    model, talker = _make_model()

    first = _forward(model, [_info(11)])
    second = _forward(model, [_info(22)])

    assert [call["marker"] for call in talker.calls] == [11, 22]
    assert first.multimodal_outputs is not None
    assert second.multimodal_outputs is not None
    assert first.multimodal_outputs["model_outputs"][0].tolist() == [11.0]
    assert second.multimodal_outputs["model_outputs"][0].tolist() == [22.0]


def test_model_intermediate_buffer_takes_precedence() -> None:
    model, talker = _make_model()

    output = _forward(model, [_info(11)], model_intermediate_buffer=[_info(22)])

    assert [call["marker"] for call in talker.calls] == [22]
    assert output.multimodal_outputs is not None
    assert output.multimodal_outputs["model_outputs"][0].tolist() == [22.0]


def test_talker_logits_preserve_batch_dimension() -> None:
    talker = MiniCPMO45OmniTTSForConditionalGeneration.__new__(MiniCPMO45OmniTTSForConditionalGeneration)

    logits = talker.compute_logits(torch.zeros(4, 768))

    assert logits.shape == (4, 2)
