# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""MiniCPM-o Live unit-audio rows: placement across steps and preemption detection."""

import pytest
import torch

from vllm_omni.config.live_session import LIVE_PREEMPTED_KEY
from vllm_omni.model_executor.models.minicpmo_4_5.live import encoder
from vllm_omni.model_executor.models.minicpmo_4_5.live.encoder import (
    LIVE_AUDIO_KEY,
    LIVE_ENCODER_KEY,
    LiveEncoderState,
    apply_live_audio,
    live_preempted_rows,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]

PLACEHOLDER = 9
HIDDEN = 4
ROWS_PER_UNIT = 2


@pytest.fixture(autouse=True)
def _stub_encoder(monkeypatch):
    def encode(thinker, state, units):
        return torch.arange(len(units) * ROWS_PER_UNIT, dtype=torch.float32).repeat_interleave(HIDDEN).view(-1, HIDDEN)

    monkeypatch.setattr(encoder, "_encode_units", encode)


def _info(seq: int, units: int, state: LiveEncoderState | None = None) -> dict:
    info = {"request_id": "r", LIVE_AUDIO_KEY: {"placeholder_id": PLACEHOLDER, "seq": seq, "units": [{}] * units}}
    if state is not None:
        info[LIVE_ENCODER_KEY] = state
    return info


def _span(text_rows: int, audio_rows: int) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.tensor([1] * text_rows + [PLACEHOLDER] * audio_rows)
    return input_ids, torch.full((input_ids.numel(), HIDDEN), -1.0)


def _step(info: dict, text_rows: int, audio_rows: int) -> tuple[torch.Tensor, LiveEncoderState]:
    applied = apply_live_audio(None, *_span(text_rows, audio_rows), info)
    assert applied is not None
    embeds, update = applied
    return embeds, update[LIVE_ENCODER_KEY]


def test_unit_rows_take_the_segment_embeddings():
    embeds, state = _step(_info(seq=1, units=2), text_rows=1, audio_rows=4)

    assert embeds[0].tolist() == [-1.0] * HIDDEN
    assert embeds[1:, 0].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert not state.preempted
    assert live_preempted_rows([_info(1, 2, state)]) == {}


def test_chunked_prefill_continues_from_the_cursor():
    _, state = _step(_info(seq=1, units=2), text_rows=0, audio_rows=2)
    embeds, state = _step(_info(seq=1, units=2, state=state), text_rows=0, audio_rows=2)

    assert embeds[:, 0].tolist() == [2.0, 3.0]
    assert not state.preempted


def test_next_segment_encodes_its_own_units():
    _, state = _step(_info(seq=1, units=1), text_rows=0, audio_rows=2)
    embeds, state = _step(_info(seq=2, units=1, state=state), text_rows=1, audio_rows=2)

    assert embeds[1:, 0].tolist() == [0.0, 1.0]
    assert not state.preempted


def test_recompute_of_earlier_units_flags_preemption():
    _, state = _step(_info(seq=1, units=1), text_rows=0, audio_rows=2)
    _, state = _step(_info(seq=2, units=1, state=state), text_rows=0, audio_rows=2)

    # After preemption the prefill re-runs both segments' rows; only the
    # current segment's embeddings are left.
    _, state = _step(_info(seq=2, units=1, state=state), text_rows=0, audio_rows=4)

    assert state.preempted
    assert live_preempted_rows([{}, _info(2, 1, state)]) == {LIVE_PREEMPTED_KEY: [False, True]}


def test_rows_without_live_state_report_nothing():
    assert live_preempted_rows([{}, {"request_id": "chat"}]) == {}
