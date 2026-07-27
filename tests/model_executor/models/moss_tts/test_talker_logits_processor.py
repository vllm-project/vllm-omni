# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for MOSS-TTS talker logits processing."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

pytest.importorskip("vllm")
pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.tts]


class _RecordingLogitsProcessor:
    def __init__(self) -> None:
        self.args: tuple[object, ...] | None = None
        self.kwargs: dict[str, object] | None = None

    def __call__(self, *args: object, **kwargs: object) -> torch.Tensor:
        self.args = args
        self.kwargs = kwargs
        return torch.zeros((1, 4))


def test_moss_tts_delay_compute_logits_does_not_forward_sampling_metadata() -> None:
    from vllm_omni.model_executor.models.moss_tts.modeling_moss_tts_talker import (
        MossTTSDelayTalkerForGeneration,
    )

    model = MossTTSDelayTalkerForGeneration.__new__(MossTTSDelayTalkerForGeneration)
    nn.Module.__init__(model)
    model.text_lm_head = object()
    model.logits_processor = _RecordingLogitsProcessor()
    model._batch_state = None
    hidden_states = torch.randn(1, 4)
    sampling_metadata = object()

    logits = model.compute_logits(hidden_states, sampling_metadata=sampling_metadata)

    assert logits.shape == (1, 4)
    assert model.logits_processor.args == (model.text_lm_head, hidden_states)
    assert model.logits_processor.kwargs == {}
