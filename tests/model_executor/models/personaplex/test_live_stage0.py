# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""PersonaPlex Live stage 0: span placement and preemption detection."""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.personaplex.live.stage0 import PersonaPlexLiveStage0, _LiveSession

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

HIDDEN = 4
SEQ = 3


def _stage0(tmp_path, *, prompt_offset: int, rows: int = 1) -> PersonaPlexLiveStage0:
    stage0 = PersonaPlexLiveStage0(SimpleNamespace(_dtype=torch.float32), model_path=str(tmp_path), max_sessions=1)
    embeds = torch.arange(rows * HIDDEN, dtype=torch.float32).view(rows, HIDDEN) + 1
    stage0.sessions["r"] = _LiveSession(slot=0, seq=SEQ, embeds=embeds, prompt_offset=prompt_offset)
    return stage0


def _preprocess(stage0: PersonaPlexLiveStage0, *, computed: int, span: int) -> torch.Tensor:
    info = {"request_id": "r", "_omni_num_computed_tokens": computed}
    _, embeds, _ = stage0.preprocess(torch.zeros(span, dtype=torch.long), {"seq": SEQ}, info)
    return embeds


def test_current_frame_span_takes_its_embeddings(tmp_path):
    stage0 = _stage0(tmp_path, prompt_offset=10)

    embeds = _preprocess(stage0, computed=10, span=1)

    assert embeds.tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert not stage0.is_preempted("r")


def test_recompute_from_before_the_frame_flags_preemption(tmp_path):
    stage0 = _stage0(tmp_path, prompt_offset=10)

    embeds = _preprocess(stage0, computed=0, span=5)

    assert embeds.shape == (5, HIDDEN)
    assert not embeds.any()
    assert stage0.is_preempted("r")
    assert not stage0.is_preempted("other")


def test_span_past_the_frame_still_raises(tmp_path):
    stage0 = _stage0(tmp_path, prompt_offset=10)

    with pytest.raises(ValueError, match="outside the prepared frame"):
        _preprocess(stage0, computed=10, span=2)

    assert not stage0.is_preempted("r")
