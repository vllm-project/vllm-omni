# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Pin the native talker's KV-recompute replay arithmetic.

Replay turns what used to be a loud desync failure into a silent
reconstruction, so these tests pin the exact index math: ``codes_rows[i]``
holds the codes sampled at step ``i + 1``, replaying step ``t`` feeds the
codes from step ``t - 1`` (``initial_code`` for ``t == 1``), the live step
(if any) is the last row of the span, and a prefill-boundary-crossing span
prepends the prefill embeds. ``build_decode_embeds`` is faked to encode
``(timeline step, prev-code sum, uncond flag)`` into each returned row so the
composition is directly assertable.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.nemotron_voicechat import talker_native
from vllm_omni.model_executor.models.nemotron_voicechat.nemotron_voicechat_talker import (
    NemotronVoiceChatTalkerForConditionalGeneration,
)

pytestmark = pytest.mark.core_model

INIT_LEN = 4
HID = 4
Q = 2


def _fake_build_decode_embeds(model, *, prev_codes, subword_embed, uncond):
    t = float(subword_embed.reshape(-1)[0])
    row = torch.tensor([[[t, float(prev_codes.float().sum()), 1.0 if uncond else 0.0, 0.0]]])
    return row


def _codes_for_step(step: int) -> torch.Tensor:
    """codes_rows entry for ``step`` (appended after the step ran)."""
    return torch.full((1, Q), 100.0 + step)


def _make_talker(step: int, rows_through_step: int):
    talker = object.__new__(NemotronVoiceChatTalkerForConditionalGeneration)
    talker._dtype = torch.float32
    talker._use_native_backbone = True
    talker.tts = SimpleNamespace(tts_model=object())
    session = {
        "use_native": True,
        "step": step,
        "init_len": INIT_LEN,
        "prefill_embeds": torch.arange(INIT_LEN * HID, dtype=torch.float32).reshape(INIT_LEN, HID),
        # cas_table[t] == [t] so the fake embed builder can recover the step.
        "cas_table": torch.arange(64, dtype=torch.float32).reshape(-1, 1),
        "timeline": torch.zeros(64, dtype=torch.long),
        "codes_rows": [_codes_for_step(s) for s in range(1, rows_through_step + 1)],
        "code": torch.full((1, 1, Q), 999.0),
        "initial_code": torch.full((1, 1, Q), 7.0),
        "silence_codes": None,
        "text_eos_id": 1,
        "uncond_stream": None,
        "uncond_hidden": None,
        "codec_streaming": True,
        "upstream_finished": False,
    }
    talker._sessions = {"req": session}
    return talker, session


def _preprocess(talker, *, offset: int, span: int):
    info = {"_omni_num_computed_tokens": offset, "request_id": "req"}
    input_ids = torch.zeros(span, dtype=torch.long)
    _, embeds, _ = talker._native_preprocess(input_ids, info, torch.device("cpu"), span, "req")
    return embeds


@pytest.fixture(autouse=True)
def fake_embed_builder(monkeypatch):
    monkeypatch.setattr(talker_native, "build_decode_embeds", _fake_build_decode_embeds)


def test_pure_replay_reads_previous_steps_codes():
    talker, session = _make_talker(step=5, rows_through_step=4)
    # Engine rewound to timeline steps 1..3 (offset init_len => t_first = 1).
    embeds = _preprocess(talker, offset=INIT_LEN, span=3)
    assert embeds.shape == (3, HID)
    assert embeds[:, 0].tolist() == [1.0, 2.0, 3.0]  # timeline steps
    # step 1 feeds initial_code; steps 2..3 feed codes sampled at steps 1..2.
    assert embeds[:, 1].tolist() == [7.0 * Q, (100.0 + 1) * Q, (100.0 + 2) * Q]
    assert embeds[:, 2].tolist() == [0.0, 0.0, 0.0]  # uncond never used
    assert session.get("pending_step") is None  # nothing to resample
    assert session["replaying"] is True


def test_replay_plus_live_step_appends_live_row_last():
    talker, session = _make_talker(step=5, rows_through_step=4)
    # Steps 3..4 replayed, step 5 is live (t_last == step).
    embeds = _preprocess(talker, offset=INIT_LEN + 2, span=3)
    assert embeds.shape == (3, HID)
    assert embeds[:, 0].tolist() == [3.0, 4.0, 5.0]
    # Replays feed stored history; the live step feeds the latest code.
    assert embeds[:, 1].tolist() == [(100.0 + 2) * Q, (100.0 + 3) * Q, 999.0 * Q]
    assert session["pending_step"] == 5
    assert "replaying" not in session


def test_replay_span_crossing_prefill_boundary_prepends_prefill():
    talker, session = _make_talker(step=4, rows_through_step=3)
    # Positions 2..3 are speaker-prompt prefill, then timeline steps 1..2.
    embeds = _preprocess(talker, offset=2, span=4)
    assert embeds.shape == (4, HID)
    assert torch.equal(embeds[:2], session["prefill_embeds"][2:4])
    assert embeds[2:, 0].tolist() == [1.0, 2.0]
    assert embeds[2:, 1].tolist() == [7.0 * Q, (100.0 + 1) * Q]
    assert session.get("pending_step") is None


def test_live_first_step_crossing_prefill_boundary_still_rejected():
    talker, _ = _make_talker(step=1, rows_through_step=0)
    with pytest.raises(ValueError, match="crossing"):
        _preprocess(talker, offset=2, span=3)


def test_short_code_history_is_rejected():
    talker, _ = _make_talker(step=5, rows_through_step=1)
    with pytest.raises(RuntimeError, match="cannot replay"):
        _preprocess(talker, offset=INIT_LEN, span=3)


def test_position_past_session_step_is_rejected():
    talker, _ = _make_talker(step=3, rows_through_step=2)
    with pytest.raises(RuntimeError, match="outpaced"):
        _preprocess(talker, offset=INIT_LEN + 3, span=1)
